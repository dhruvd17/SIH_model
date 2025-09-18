import os
import xgboost as xgb
import pandas as pd
import numpy as np
import gradio as gr
import json
from PIL import Image
from scipy.ndimage import uniform_filter

# To handle potentially large DEM files with Pillow
Image.MAX_IMAGE_PIXELS = None

# ----- config -----
# Define band indices for NDVI calculation. This might need to be adjusted based on the specific satellite/drone source.
# For this example, we assume the GeoTIFFs provided have Red as the first band and NIR as the second.
RED_BAND_INDEX = 1
NIR_BAND_INDEX = 2

# Load features and label map from the model config file
try:
    with open('rockfall_xgb.json', 'r') as f:
        model_config = json.load(f)
    FEATURES = model_config['learner']['feature_names']
    LABEL_MAP = {0: "Low", 1: "Medium", 2: "High", 3: "Extreme"}
    print("Model config loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load or parse rockfall_xgb.json: {e}. Using fallback features.")
    FEATURES = [
        "aspect","clay",
        "contextual_slope_1000m","contextual_slope_300m","contextual_slope_5000m",
        "curvature","elevation","hand",
        "ndvi_change","ndvi_post","ndvi_pre",
        "organicC","rainfall","relief","ruggedness",
        "sand","silt","twi"
    ]
    LABEL_MAP = {0: "Low", 1: "Medium", 2: "High", 3: "Extreme"}


# ----- load model safely (json) -----
try:
    model = xgb.XGBClassifier()
    model.load_model('rockfall_xgb.json')
    print("XGBoost model loaded successfully from JSON.")
except Exception as e:
    print(f"Error loading XGBoost model from JSON: {e}")
    model = None


# ----- helpers -----
def _predict_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Makes predictions using the loaded XGBoost model on a DataFrame.
    """
    if model is None:
        raise RuntimeError("XGBoost model not loaded.")

    # Ensure feature order and types, fill missing columns with NaN
    X = df.reindex(columns=FEATURES, fill_value=np.nan)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    out = df.copy()
    out["predicted_class"] = preds.astype(int)
    out["predicted_label"] = [LABEL_MAP.get(int(c), str(int(c))) for c in preds]

    # add per-class probabilities
    for i in range(probs.shape[1]):
        label = LABEL_MAP.get(i, str(i))
        out[f"prob_{label}"] = probs[:, i]

    return out

# ----- Data Processing Functions for File Analysis (using Pillow and NumPy) -----

def process_dem_file(dem_path):
    """
    Processes a Digital Elevation Model (DEM) file using Pillow and NumPy
    to calculate various terrain attributes.
    """
    print(f"Analyzing DEM file: {dem_path}")
    try:
        with Image.open(dem_path) as img:
            dem_data = np.array(img).astype(float)
        
        # Assume a common no-data value and replace it with NaN
        no_data_val = -9999
        dem_data[dem_data == no_data_val] = np.nan

        # Calculate gradients
        dy, dx = np.gradient(dem_data)

        # Slope
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

        # Aspect
        aspect = np.degrees(np.arctan2(-dy, dx))
        aspect[aspect < 0] += 360

        # Curvature (simplified as the Laplacian)
        dxx, _ = np.gradient(dx)
        dyy, _ = np.gradient(dy)
        curvature = dxx + dyy

        # Ruggedness (Terrain Ruggedness Index - TRI)
        squared_diff = np.zeros_like(dem_data)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                shifted = np.roll(dem_data, shift=(i, j), axis=(0, 1))
                squared_diff += (dem_data - shifted)**2
        ruggedness = np.sqrt(squared_diff)
        
        # Contextual Slopes (approximated by smoothing the DEM first)
        def get_contextual_slope(data, size):
            smoothed_dem = uniform_filter(data, size=size, mode='reflect')
            s_dy, s_dx = np.gradient(smoothed_dem)
            return np.degrees(np.arctan(np.sqrt(s_dx**2 + s_dy**2)))

        # Assuming 30m resolution: 300m is ~11 pixels, 1000m is ~33, etc.
        slope_300m = get_contextual_slope(dem_data, 11)
        slope_1000m = get_contextual_slope(dem_data, 33)
        slope_5000m = get_contextual_slope(dem_data, 167)
        
        # Global relief
        relief = np.nanmax(dem_data) - np.nanmin(dem_data)

        # Hydrological features are too complex for a simple NumPy implementation.
        # Return NaN and let the model handle them.
        twi = np.nan
        hand = np.nan

        return {
            'elevation': np.nanmean(dem_data),
            'aspect': np.nanmean(aspect),
            'curvature': np.nanmean(curvature),
            'relief': relief,
            'ruggedness': np.nanmean(ruggedness),
            'contextual_slope_300m': np.nanmean(slope_300m),
            'contextual_slope_1000m': np.nanmean(slope_1000m),
            'contextual_slope_5000m': np.nanmean(slope_5000m),
            'twi': twi,
            'hand': hand
        }
    except Exception as e:
        print(f"Error processing DEM file {dem_path}: {e}")
        return {k: np.nan for k in ['elevation', 'aspect', 'curvature', 'relief', 'ruggedness', 'contextual_slope_300m', 'contextual_slope_1000m', 'contextual_slope_5000m', 'twi', 'hand']}

def process_ndvi_files(image_pre_path, image_post_path):
    """
    Processes two satellite/drone images to calculate NDVI change using Pillow.
    """
    print(f"Analyzing NDVI 'before' image: {image_pre_path}")
    print(f"Analyzing NDVI 'after' image: {image_post_path}")

    def calculate_mean_ndvi(image_path):
        with Image.open(image_path) as img:
            bands = img.split()
            if len(bands) < NIR_BAND_INDEX:
                raise ValueError(f"Image {image_path} does not have enough bands for NDVI calculation.")
            red = np.array(bands[RED_BAND_INDEX - 1]).astype(float)
            nir = np.array(bands[NIR_BAND_INDEX - 1]).astype(float)
        
        np.seterr(divide='ignore', invalid='ignore')
        ndvi_map = (nir - red) / (nir + red)
        return np.nanmean(ndvi_map)
    try:
        ndvi_pre = calculate_mean_ndvi(image_pre_path)
        ndvi_post = calculate_mean_ndvi(image_post_path)
        return {
            'ndvi_pre': ndvi_pre,
            'ndvi_post': ndvi_post,
            'ndvi_change': ndvi_post - ndvi_pre
        }
    except Exception as e:
        print(f"Error processing NDVI files: {e}")
        return {'ndvi_pre': np.nan, 'ndvi_post': np.nan, 'ndvi_change': np.nan}

# ----- Main Prediction Logic -----
def predict_rockfall_risk(dem_file, image_pre_file, image_post_file, csv_file):
    """
    Predicts rockfall risk by processing user-provided files and soil/weather data from a CSV file.
    CSV must contain columns: rainfall, clay, sand, silt, organicC
    """
    if not all([dem_file, image_pre_file, image_post_file, csv_file]):
        return "Error: DEM, NDVI files, and CSV file are required.", None

    try:
        all_features = {}

        # Process DEM
        dem_features = process_dem_file(dem_file.name)
        all_features.update(dem_features)

        # Process NDVI
        ndvi_features = process_ndvi_files(image_pre_file.name, image_post_file.name)
        all_features.update(ndvi_features)

        # Read soil/environmental data from CSV
        df_csv = pd.read_csv(csv_file.name)
        required_cols = ["rainfall", "clay", "sand", "silt", "organicC"]
        for col in required_cols:
            if col not in df_csv.columns:
                raise ValueError(f"CSV missing required column: {col}")

        # Assuming first row has the values
        all_features.update(df_csv.iloc[0][required_cols].to_dict())

        # Predict
        input_df = pd.DataFrame([all_features])
        scored_df = _predict_df(input_df)

        predicted_label = scored_df.loc[0, "predicted_label"]
        prob_cols = [c for c in scored_df.columns if c.startswith("prob_")]
        probs_df = scored_df.loc[0, prob_cols].rename_axis("class").reset_index(name="probability")
        probs_df["class"] = probs_df["class"].str.replace("prob_", "", regex=False)
        probs_df = probs_df.sort_values("probability", ascending=False)
        probs_df['probability'] = probs_df['probability'].map('{:.2%}'.format)

        return predicted_label, probs_df

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return f"Error: {e}", None


# ----- Gradio UI -----
with gr.Blocks(title="⛏️ Rockfall Risk Prediction") as demo:
    gr.Markdown("# ⛏️ Rockfall Risk Prediction")
    gr.Markdown("Upload the required geospatial data files and a CSV file with soil/weather data.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Terrain & Topography Data")
            dem_input = gr.File(label="Upload Digital Elevation Model (DEM) File")

            gr.Markdown("### 3. Soil & Environmental Data (CSV)")
            csv_input = gr.File(label="Upload CSV File (rainfall, clay, sand, silt, organicC)", file_types=[".csv"])

        with gr.Column(scale=1):
            gr.Markdown("### 2. Vegetation Data - Before")
            image_pre_input = gr.File(label="Upload 'Before' Satellite/Drone Image (with NIR band)")
            
            gr.Markdown("### 2. Vegetation Data - After")
            image_post_input = gr.File(label="Upload 'After' Satellite/Drone Image (with NIR band)")
        
    with gr.Row():
        predict_button = gr.Button("Predict Rockfall Risk", variant="primary")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📈 Prediction Results")
            predicted_label_output = gr.Label(label="Predicted Risk Level")
        with gr.Column(scale=2):
            probability_table_output = gr.Dataframe(headers=["class", "probability"], label="Risk Probabilities")

    predict_button.click(
        predict_rockfall_risk,
        inputs=[dem_input, image_pre_input, image_post_input, csv_input],
        outputs=[predicted_label_output, probability_table_output]
    )

if __name__ == "__main__":
    demo.launch(debug=True)

