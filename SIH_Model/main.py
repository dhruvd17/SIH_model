from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image
import xgboost as xgb

MODEL_PATH = "rockfall_xgb.json"
bst = None  # global model

# ========== Request Schema ==========
class InputFiles(BaseModel):
    image1: str
    image2: str
    demFile: str
    csvFile: str

# ========== NDVI Calculation ==========
def calculate_ndvi(image_bytes: bytes) -> float:
    try:
        img = Image.open(BytesIO(image_bytes))
        bands = img.split()

        # single-band image case
        if len(bands) == 1:
            arr = np.array(bands[0], dtype="float32")
            return float(arr.mean())  # fallback intensity

        # multi-band (assume band1=RED, band2=NIR)
        red = np.array(bands[0], dtype="float32")
        nir = np.array(bands[1], dtype="float32")
        ndvi = (nir - red) / (nir + red + 1e-6)
        return float(np.nanmean(ndvi))
    except Exception as e:
        print(f"NDVI calc error: {e}")
        return float("nan")

# ========== Feature Extractor ==========
def extract_features(files: InputFiles):
    try:
        # download CSV with timeout
        try:
            resp = requests.get(files.csvFile, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"CSV fetch error: {e}")
            raise HTTPException(400, f"CSV download failed: {e}")

        df = pd.read_csv(BytesIO(resp.content))
        df.columns = df.columns.str.strip().str.lower()

        req_cols = ["rainfall", "clay", "sand", "silt", "organicc"]
        for c in req_cols:
            if c not in df.columns:
                raise HTTPException(400, f"CSV missing column: {c}")

        features = {c: float(df[c].mean()) for c in req_cols}

        # download & process image1
        try:
            r1 = requests.get(files.image1, timeout=15)
            r1.raise_for_status()
            features["ndvi_pre"] = calculate_ndvi(r1.content)
        except Exception as e:
            print(f"Image1 error: {e}")
            features["ndvi_pre"] = float("nan")

        # download & process image2
        try:
            r2 = requests.get(files.image2, timeout=15)
            r2.raise_for_status()
            features["ndvi_post"] = calculate_ndvi(r2.content)
        except Exception as e:
            print(f"Image2 error: {e}")
            features["ndvi_post"] = float("nan")

        # NDVI change
        if not np.isnan(features["ndvi_pre"]) and not np.isnan(features["ndvi_post"]):
            features["ndvi_change"] = features["ndvi_post"] - features["ndvi_pre"]
        else:
            features["ndvi_change"] = float("nan")

        return features
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        raise HTTPException(400, f"Feature extraction failed: {e}")

# ========== FastAPI App ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global bst
    try:
        bst = xgb.Booster()
        bst.load_model(MODEL_PATH)
        print("✅ XGBoost model loaded")
    except Exception as e:
        print(f"❌ Model load error: {e}")
        bst = None
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/links-predict")
def predict_from_links(files: InputFiles):
    if bst is None:
        raise HTTPException(500, "Model not loaded")

    features = extract_features(files)
    df = pd.DataFrame([features])

    dmatrix = xgb.DMatrix(df)
    preds = bst.predict(dmatrix)
    pred_label = int(preds[0] >= 0.5)

    return {"prediction": pred_label, "probability": float(preds[0])}