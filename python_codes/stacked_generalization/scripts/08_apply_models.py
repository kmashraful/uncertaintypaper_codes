import os
import joblib
import warnings
import rasterio
import numpy as np
import xgboost as xgb
from glob import glob
from osgeo import gdal
from tqdm import tqdm

warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")


def create_vrt(input_rasters, output_vrt):
    """
    Create a VRT using GDAL from a list of input raster paths.
    Use average resampling and do not separate bands.
    """
    vrt_options = gdal.BuildVRTOptions(
        separate = False,
        resampleAlg = 'average',
        resolution = 'highest',
        srcNodata = -32_768,
        VRTNodata  = -32_768
        )
    gdal.BuildVRT(output_vrt, input_rasters, options=vrt_options)

    return None


def apply_model_window(
    data_window, 
    model, 
    scaler, 
    nodata_output = -32_768, 
    nodata_value = -32_768
    ):
    """
    Apply the model to a 3D window of shape (bands, rows, cols).
    Returns a 2D array (rows, cols) in float32, using 'nodata_output' for 
    pixels containing NaNs or the specified 'nodata_value'.
    """
    bands, rows, cols = data_window.shape
    
    # Flatten data to (rows*cols, bands) for model input
    reshaped = data_window.reshape(bands, -1).T
    
    # Identify pixels that are NaN in any band
    nan_mask = np.any(np.isnan(reshaped), axis=1)
    
    # Identify pixels that match the no-data value in any band
    nodata_mask = np.any(data_window == nodata_value, axis=0).ravel()
    
    # Combine masks so that any NaN or no-data pixel is excluded
    exclude_mask = nan_mask | nodata_mask
    
    # Initialize the prediction array
    predictions = np.full((rows * cols,), nodata_output, dtype=np.float32)
    
    # Perform inference on valid pixels only
    valid_pixels = ~exclude_mask
    if np.any(valid_pixels):
        valid_data = reshaped[valid_pixels]
        valid_data_scaled = scaler.transform(valid_data)
        preds = model.predict_proba(valid_data_scaled)[:, 1]
        
        # Clip predictions to 0-255
        preds_scaled = np.clip(preds, 0, 255).astype(np.float32)
        predictions[valid_pixels] = preds_scaled
    
    # Reshape predictions back to (rows, cols)
    return predictions.reshape((rows, cols))

    
def predict_and_write_mosaic(vrt_path, output_tif, model, scaler, nodata_output=-32_768):
    """
    Opens the VRT, iterates over blocks, applies the model with 'apply_model_window',
    and writes results to 'output_tif' as a single-band byte raster.
    """
    with rasterio.open(vrt_path) as src:
        meta = src.meta.copy()
        meta.update({
            'driver': 'GTiff',
            'count': 1,
            'dtype': 'float32',
            'nodata': nodata_output,
            'compress': 'lzw'
        })
        
        # Convert block_windows to a list to determine its length
        windows = list(src.block_windows(1))
        
        with rasterio.open(output_tif, 'w', **meta) as dst, tqdm(total=len(windows), desc="Applying model to blocks") as pbar:
            for idx, window in windows:
                data = src.read(window=window)
                result_window = apply_model_window(data, model, scaler, nodata_output=nodata_output)
                dst.write(result_window, 1, window=window)
                pbar.update(1)

    return None


if __name__ == "__main__":
    
    # Define the folder where models
    models_folder = "C:/Users/johnb/Documents/git/masters_project/stacked_generalization/models/"
    baselearner_folder = f"{models_folder}/baselearners"
    
    # Define the folder where the predictions should be stored
    predictions_folder = "C:/Users/johnb/Desktop/sundarbans_cover_maps"
    output_folder = f"{predictions_folder}/outputs"
    if not os.path.exists(predictions_folder):
        os.mkdir(predictions_folder)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Read in the Raster dataset
    raster_paths = glob("C:/Users/johnb/Desktop/Mangrove_Features/*.tif")
    
    # Define where the VRT will be saved
    vrt_path = "C:/Users/johnb/Desktop/sundarbans_cover_maps/vrts/planet_features.vrt"
    
    ###########################################################################
    ### STEP 1. GENERATE A VRT FROM THE FEATURES DATASET
    ###########################################################################
    
    # Generate the VRT
    create_vrt(raster_paths, vrt_path)
    
    ###########################################################################
    ### STEP 2. GENERATE BASELEARNER MAPS
    ###########################################################################
    
    # Load the standard scaler
    scaler = joblib.load(f"{models_folder}/scaler.joblib")
    
    # Load the base learner models
    baselearner_knn_model = joblib.load(f"{baselearner_folder}/baselearner_knn_model.joblib")
    baselearner_logreg_model = joblib.load(f"{baselearner_folder}/baselearner_logreg_model.joblib")
    baselearner_rf_model = joblib.load(f"{baselearner_folder}/baselearner_rf_model.joblib")
    baselearner_svc_model = joblib.load(f"{baselearner_folder}/baselearner_svc_model.joblib")
    baselearner_xgb_model = joblib.load(f"{baselearner_folder}/baselearner_xgb_model.joblib")

    # Load the stacking models
    stacking_logreg_npt_model = joblib.load(f"{baselearner_folder}/stacking_logreg_npt.joblib")
    stacking_logreg_pt_model = joblib.load(f"{baselearner_folder}/stacking_logreg_np.joblib")
    stacking_rf_npt_model = joblib.load(f"{baselearner_folder}/stacking_rf_npt.joblib")
    stacking_rf_pt_model = joblib.load(f"{baselearner_folder}/stacking_rf_np.joblib")
    
    # Apply the KNN model
    output_path = os.path.join(output_folder, "baselearner_knn_prediction.tif")
    predict_and_write_mosaic(vrt_path, output_path, baselearner_knn_model, scaler, nodata_output=-32_768)
        
    # Apply the Logistic Regression model
    output_path = os.path.join(output_folder, "baselearner_logreg_prediction.tif")
    predict_and_write_mosaic(vrt_path, output_path, baselearner_logreg_model, scaler, nodata_output=-32_768)
        
    # Apply the Random Forests model
    output_path = os.path.join(output_folder, "baselearner_rf_prediction.tif")
    predict_and_write_mosaic(vrt_path, output_path, baselearner_rf_model, scaler, nodata_output=-32_768)

    # Apply the Support Vector Machine model
    output_path = os.path.join(output_folder, "baselearner_svc_prediction.tif")
    predict_and_write_mosaic(vrt_path, output_path, baselearner_svc_model, scaler, nodata_output=-32_768)
                        
    # Apply the XGB model
    output_path = os.path.join(output_folder, "baselearner_xgb_prediction.tif")
    predict_and_write_mosaic(vrt_path, output_path, baselearner_xgb_model, scaler, nodata_output=-32_768)
        
    # Apply the stacking w/ no passthrough and log regression stacking
    output_path = os.path.join(output_folder, "stacking_logreg_npt_prediction.tif")
    predict_and_write_mosaic(vrt_path, output_path, stacking_logreg_npt_model, scaler, nodata_output=-32_768)
    
    # Apply the stacking w/ passthrough and log regression stacking
    output_path = os.path.join(output_folder, "stacking_logreg_pt_prediction.tif")
    predict_and_write_mosaic(vrt_path, output_path, stacking_logreg_pt_model, scaler, nodata_output=-32_768)
    
    # Apply the stacking w/out passthrough and random forest stacking
    output_path = os.path.join(output_folder, "stacking_rf_npt_prediction.tif")
    predict_and_write_mosaic(vrt_path, output_path, stacking_rf_npt_model, scaler, nodata_output=-32_768)
    
    # Apply the stacking w/ passthrough and random forest stacking
    output_path = os.path.join(output_folder, "stacking_rf_pt_prediction.tif")
    predict_and_write_mosaic(vrt_path, output_path, stacking_rf_pt_model, scaler, nodata_output=-32_768)


