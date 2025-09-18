# main.py
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "rockfall_xgb.json"

# Global model handle (can be XGBClassifier or Booster)
bst = None

# Define the request schema matching your feature names
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bst
    # Try sklearn API first
    try:
        clf = xgb.XGBClassifier()
        clf.load_model(MODEL_PATH)  # must match the way the model was saved
        bst = clf
        print("Loaded XGBClassifier model.")
    except Exception as e1:
        print("XGBClassifier load failed:", e1)
        # Fallback to native Booster
        try:
            booster = xgb.Booster()
            booster.load_model(MODEL_PATH)
            bst = booster
            print("Loaded Booster model.")
        except Exception as e2:
            print("Booster load failed:", e2)
            bst = None
    yield
    # Optional: cleanup on shutdown

app = FastAPI(title="Rockfall Risk API", lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(item: Features):
    if bst is None:
        raise HTTPException(500, "XGBoost model not loaded")

    # Build DataFrame from request and coerce to numeric
    X = pd.DataFrame([item.model_dump()])
    # Force numeric types; None/invalid -> NaN which XGBoost supports
    X = X.apply(pd.to_numeric, errors="coerce").astype("float64")

    try:
        # Path 1: sklearn API (XGBClassifier/Regressor)
        if hasattr(bst, "predict_proba"):
            proba = bst.predict_proba(X)[0].tolist()
            pred = int(np.argmax(proba))
            return {"pred": pred, "proba": proba}

        # Path 2: native Booster
        dmat = xgb.DMatrix(X.values)  # NaNs allowed by default
        proba = bst.predict(dmat)[0].tolist()
        pred = int(np.argmax(proba))
        return {"pred": pred, "proba": proba}
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {e}")
