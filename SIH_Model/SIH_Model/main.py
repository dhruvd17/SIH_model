# main.py
from contextlib import asynccontextmanager
from typing import Optional, Tuple, Dict
import io
import numpy as np
import pandas as pd
import xgboost as xgb
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from PIL import Image
from scipy.ndimage import uniform_filter
from fastapi.middleware.cors import CORSMiddleware  # optional, configure below

# Allow very large rasters
Image.MAX_IMAGE_PIXELS = None

# ---------- Config ----------
MODEL_PATH = "rockfall_xgb.json"
LABELS = ["Low", "Medium", "High", "Extreme"]

# NDVI band indices (1-based)
RED_BAND_INDEX = 1
NIR_BAND_INDEX = 2

# ---------- Model load ----------
bst = None  # XGBClassifier or Booster

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bst
    try:
        clf = xgb.XGBClassifier()
        clf.load_model(MODEL_PATH)
        bst = clf
        print("Loaded XGBClassifier model.")
    except Exception as e1:
        print("XGBClassifier load failed:", e1)
        try:
            booster = xgb.Booster()
            booster.load_model(MODEL_PATH)
            bst = booster
            print("Loaded Booster model.")
        except Exception as e2:
            print("Booster load failed:", e2)
            bst = None
    yield

app = FastAPI(title="Rockfall Risk API", lifespan=lifespan)

# Optional CORS for a browser frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schemas ----------
class Features(BaseModel):
    aspect: Optional[float] = None
    clay: Optional[float] = None
    contextual_slope_1000m: Optional[float] = None
    contextual_slope_300m: Optional[float] = None
    contextual_slope_5000m: Optional[float] = None
    curvature: Optional[float] = None
    elevation: Optional[float] = None
    hand: Optional[float] = None
    ndvi_change: Optional[float] = None
    ndvi_post: Optional[float] = None
    ndvi_pre: Optional[float] = None
    organicC: Optional[float] = None
    rainfall: Optional[float] = None
    relief: Optional[float] = None
    ruggedness: Optional[float] = None
    sand: Optional[float] = None
    silt: Optional[float] = None
    twi: Optional[float] = None

class LinkRequest(BaseModel):
    image1: HttpUrl
    image2: HttpUrl
    demFile: HttpUrl
    csvFile: HttpUrl

# ---------- Helpers ----------
def _score_dataframe(X: pd.DataFrame) -> Tuple[int, np.ndarray]:
    if bst is None:
        raise HTTPException(500, "XGBoost model not loaded")
    try:
        if hasattr(bst, "predict_proba"):
            proba = bst.predict_proba(X)[0]
        else:
            dmat = xgb.DMatrix(X.values)
            proba = bst.predict(dmat)[0]
        idx = int(np.argmax(proba))
        return idx, np.asarray(proba, dtype=float)
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {e}")

def _label_conf(proba: np.ndarray) -> Tuple[str, float]:
    idx = int(np.argmax(proba))
    label = LABELS[idx] if idx < len(LABELS) else str(idx)
    conf = float(proba[idx]) * 100.0
    return label, round(conf, 2)

def process_dem_image_pillow(dem_img: Image.Image) -> Dict[str, float]:
    try:
        dem = np.array(dem_img).astype(float)
        no_data_val = -9999
        dem[dem == no_data_val] = np.nan

        dy, dx = np.gradient(dem)
        aspect = np.degrees(np.arctan2(-dy, dx))
        aspect[aspect < 0] += 360

        dxx, _ = np.gradient(dx)
        dyy, _ = np.gradient(dy)
        curvature = dxx + dyy

        # Simple ruggedness proxy
        sq = np.zeros_like(dem)
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i == 0 and j == 0:
                    continue
                shifted = np.roll(dem, shift=(i, j), axis=(0, 1))
                sq += (dem - shifted) ** 2
        ruggedness = np.sqrt(sq)

        def ctx_slope(data, size):
            s_dem = uniform_filter(data, size=size, mode="reflect")
            s_dy, s_dx = np.gradient(s_dem)
            return np.degrees(np.arctan(np.sqrt(s_dx**2 + s_dy**2)))

        slope_300m = ctx_slope(dem, 11)
        slope_1000m = ctx_slope(dem, 33)
        slope_5000m = ctx_slope(dem, 167)

        relief = float(np.nanmax(dem) - np.nanmin(dem))

        return {
            "elevation": float(np.nanmean(dem)),
            "aspect": float(np.nanmean(aspect)),
            "curvature": float(np.nanmean(curvature)),
            "relief": relief,
            "ruggedness": float(np.nanmean(ruggedness)),
            "contextual_slope_300m": float(np.nanmean(slope_300m)),
            "contextual_slope_1000m": float(np.nanmean(slope_1000m)),
            "contextual_slope_5000m": float(np.nanmean(slope_5000m)),
            "twi": np.nan,
            "hand": np.nan,
        }
    except Exception:
        keys = [
            "elevation","aspect","curvature","relief","ruggedness",
            "contextual_slope_300m","contextual_slope_1000m","contextual_slope_5000m",
            "twi","hand"
        ]
        return {k: np.nan for k in keys}

def compute_mean_ndvi(img: Image.Image) -> float:
    bands = img.split()
    if len(bands) < max(RED_BAND_INDEX, NIR_BAND_INDEX):
        return float("nan")
    red = np.array(bands[RED_BAND_INDEX - 1]).astype(float)
    nir = np.array(bands[NIR_BAND_INDEX - 1]).astype(float)
    np.seterr(divide="ignore", invalid="ignore")
    ndvi = (nir - red) / (nir + red)
    return float(np.nanmean(ndvi))

def process_ndvi_pair(img1: Image.Image, img2: Image.Image) -> Dict[str, float]:
    try:
        ndvi_pre = compute_mean_ndvi(img1)
        ndvi_post = compute_mean_ndvi(img2)
        return {
            "ndvi_pre": ndvi_pre,
            "ndvi_post": ndvi_post,
            "ndvi_change": float(ndvi_post - ndvi_pre) if np.isfinite(ndvi_pre) and np.isfinite(ndvi_post) else float("nan"),
        }
    except Exception:
        return {"ndvi_pre": np.nan, "ndvi_post": np.nan, "ndvi_change": np.nan}

# ---------- Routes ----------
@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(item: Features):
    X = pd.DataFrame([item.model_dump()]).apply(pd.to_numeric, errors="coerce").astype("float64")
    idx, proba = _score_dataframe(X)
    label, conf = _label_conf(proba)
    return {"danger": label, "confidence": conf, "proba": proba.tolist()}

@app.post("/links-predict")
def links_predict(req: LinkRequest):
    # 1) Download inputs
    try:
        img1_b = requests.get(str(req.image1), timeout=30).content
        img2_b = requests.get(str(req.image2), timeout=30).content
        dem_b  = requests.get(str(req.demFile), timeout=60).content
        csv_b  = requests.get(str(req.csvFile), timeout=30).content
    except Exception as e:
        raise HTTPException(400, f"Download failed: {e}")

    # 2) Feature extraction with Pillow/NumPy and pandas
    try:
        with Image.open(io.BytesIO(dem_b)) as dem_img:
            dem_img.load()
            dem_feats = process_dem_image_pillow(dem_img)

        with Image.open(io.BytesIO(img1_b)) as img1:
            img1.load()
        with Image.open(io.BytesIO(img2_b)) as img2:
            img2.load()
            ndvi_feats = process_ndvi_pair(img1, img2)

        df_csv = pd.read_csv(io.BytesIO(csv_b))
        req_cols = ["rainfall", "clay", "sand", "silt", "organicC"]
        for c in req_cols:
            if c not in df_csv.columns:
                raise ValueError(f"CSV missing required column: {c}")
        csv_feats = df_csv.iloc[0][req_cols].to_dict()
        csv_feats = {k: float(csv_feats[k]) for k in req_cols}
    except Exception as e:
        raise HTTPException(400, f"Feature extraction failed: {e}")

    # 3) Score
    all_features = {}
    all_features.update(dem_feats)
    all_features.update(ndvi_feats)
    all_features.update(csv_feats)

    X = pd.DataFrame([all_features]).apply(pd.to_numeric, errors="coerce").astype("float64")
    idx, proba = _score_dataframe(X)
    label, conf = _label_conf(proba)
    return {"danger": label, "confidence": conf, "proba": proba.tolist()}
