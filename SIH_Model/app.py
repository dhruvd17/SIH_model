import os
import xgboost as xgb
import pandas as pd
import numpy as np
import gradio as gr
import json
from PIL import Image
from scipy.ndimage import uniform_filter

Image.MAX_IMAGE_PIXELS = None

RED_BAND_INDEX = 1
NIR_BAND_INDEX = 2

# ----- load model -----
try:
    model = xgb.XGBClassifier()
    model.load_model("rockfall_xgb.json")
    FEATURES = model.get_booster().feature_names
    LABEL_MAP = {0: "Low", 1: "Medium", 2: "High", 3: "Extreme"}
    print("✅ XGBoost model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    FEATURES = []
    LABEL_MAP = {0: "Low", 1: "Medium", 2: "High", 3: "Extreme"}

# ----- helpers -----
def _predict_df(df: pd.DataFrame) -> pd.DataFrame:
    if model is None:
        raise RuntimeError("XGBoost model not loaded")
    X = df.reindex(columns=FEATURES, fill_value=np.nan)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    out = df.copy()
    out["predicted_label"] = [LABEL_MAP.get(int(c), str(c)) for c in preds]
    for i in range(probs.shape[1]):
        out[f"prob_{LABEL_MAP.get(i, str(i))}"] = probs[:, i]
    return out

def process_dem_file(path):
    try:
        with Image.open(path) as img:
            dem = np.array(img).astype(float)
        dem[dem == -9999] = np.nan
        dy, dx = np.gradient(dem)
        aspect = np.degrees(np.arctan2(-dy, dx))
        aspect[aspect < 0] += 360
        dxx, _ = np.gradient(dx)
        dyy, _ = np.gradient(dy)
        curvature = dxx + dyy
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
            return np.degrees(np.arctan(np.sqrt(s_dx*2 + s_dy*2)))
        slope_300m = ctx_slope(dem, 11)
        slope_1000m = ctx_slope(dem, 33)
        slope_5000m = ctx_slope(dem, 167)
        relief = np.nanmax(dem) - np.nanmin(dem)
        return {
            "elevation": np.nanmean(dem),
            "aspect": np.nanmean(aspect),
            "curvature": np.nanmean(curvature),
            "relief": relief,
            "ruggedness": np.nanmean(ruggedness),
            "contextual_slope_300m": np.nanmean(slope_300m),
            "contextual_slope_1000m": np.nanmean(slope_1000m),
            "contextual_slope_5000m": np.nanmean(slope_5000m),
            "twi": np.nan,
            "hand": np.nan,
        }
    except Exception as e:
        print("DEM processing error:", e)
        return {k: np.nan for k in ["elevation","aspect","curvature","relief","ruggedness","contextual_slope_300m","contextual_slope_1000m","contextual_slope_5000m","twi","hand"]}

def process_ndvi_files(pre, post):
    def calc_ndvi(path):
        try:
            with Image.open(path) as img:
                bands = img.split()
                if len(bands) < max(RED_BAND_INDEX, NIR_BAND_INDEX):
                    return np.nan
                red = np.array(bands[RED_BAND_INDEX-1]).astype(float)
                nir = np.array(bands[NIR_BAND_INDEX-1]).astype(float)
                ndvi = (nir - red) / (nir + red)
                return np.nanmean(ndvi)
        except Exception:
            return np.nan
    ndvi_pre = calc_ndvi(pre)
    ndvi_post = calc_ndvi(post)
    ndvi_change = ndvi_post - ndvi_pre if np.isfinite(ndvi_pre) and np.isfinite(ndvi_post) else np.nan
    return {"ndvi_pre": ndvi_pre, "ndvi_post": ndvi_post, "ndvi_change": ndvi_change}

def predict_rockfall_risk(dem_file, pre_img, post_img, csv_file):
    try:
        feats = {}
        feats.update(process_dem_file(dem_file.name))
        feats.update(process_ndvi_files(pre_img.name, post_img.name))
        df_csv = pd.read_csv(csv_file.name)
        required_cols = ["rainfall","clay","sand","silt","organicC"]
        feats.update(df_csv.iloc[0][required_cols].astype(float).to_dict())
        input_df = pd.DataFrame([feats])
        scored = _predict_df(input_df)
        label = scored.loc[0,"predicted_label"]
        probs = [ (c.replace("prob_",""), f"{scored.loc[0,c]*100:.2f}%") for c in scored.columns if c.startswith("prob_")]
        return label, probs
    except Exception as e:
        return f"Error: {e}", []

# ----- Gradio UI -----
with gr.Blocks(title="⛏ Rockfall Risk Prediction") as demo:
    gr.Markdown("# ⛏ Rockfall Risk Prediction")
    with gr.Row():
        with gr.Column():
            dem_input = gr.File(label="DEM File")
            csv_input = gr.File(label="CSV File", file_types=[".csv"])
        with gr.Column():
            pre_input = gr.File(label="NDVI Before Image")
            post_input = gr.File(label="NDVI After Image")
    predict_btn = gr.Button("Predict", variant="primary")
    label_out = gr.Label(label="Predicted Risk Level")
    probs_out = gr.Dataframe(headers=["Class","Probability"], label="Probabilities")
    predict_btn.click(predict_rockfall_risk, inputs=[dem_input,pre_input,post_input,csv_input], outputs=[label_out,probs_out])

if _name_ == "_main_":
    demo.launch(debug=True)