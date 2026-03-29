"""
Script Name: 03_format_modeling_datasets.py
Author: John Kilbride
Date: 2025-03-16
Description: 
    
    Prepares the modeling datasets used for hyperparameter tuning and for validating
    the final model performance. 
    
    A single 5-fold CV dataset is used for hyperparameter tuning
    
    100 repeated 5-fold CV datasets are used to estimate the final accuracy. 
    
    The script saves the the tuning dataset, the formatted repeated CV datasets, 
    and the standard scaler objects used for compute the mean and standard deviation. 
    
"""
import os
import joblib
import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    
    # Define the folder where outputs will be stored
    intermediate_folder = "C:/Users/johnb/Documents/git/masters_project/stacked_generalization/intermediate_outputs"
    
    # Define the folder where models
    models_folder = "C:/Users/johnb/Documents/git/masters_project/stacked_generalization/models"

    ###########################################################################
    ### STEP 1. FORMAT HYPERPARAMETER TUNING DATASET
    ###########################################################################
    
    # Load in the modeling dataframe
    training_df = pd.read_csv(f"{intermediate_folder}/raw_tuning_dataset.csv")
    
    # Re-code 'landcover' column into 1 for "mangrove" and 0 for "non-mangrove"
    training_df['landcover'] = training_df['landcover'].map({'mangrove': 1, 'nonmangrove': 0})
    
    # Define all of the features
    features = [
        'ARVI', 'ARVI_INTP', 'B', 'CMRI', 'CMRI_INTP', 'CVI', 'CVI_INTP',
        'EVI_INTP', 'EVI_RMSE', 'G', 'GLI', 'GLI_INTP', 'IPVI', 'IPVI_INTP', 
        'MCARI1', 'MCARI1_INTP', 'MCARI2', 'MCARI2_INTP', 'N', 'NDTI',
        'NDTI_INTP', 'NDVI', 'NDVI_INTP', 'NDWI', 'NDWI_INTP', 'NLI',
        'NLI_INTP', 'N_INTP', 'OSAVI', 'OSAVI_INTP', 'R', 'R_INTP', 'TDVI',
        'TDVI_INTP', 'TVI', 'TVI_INTP', 'TriVI', 'TriVI_INTP', 'WDRVI', 'WDRVI_INTP'
        ]
    
    # Initialize the StandardScaler
    scaler = StandardScaler()
    training_df[features] = scaler.fit_transform(training_df[features])
    
    # Save the training dataset
    training_df.to_csv(f"{intermediate_folder}/tuning_dataset.csv")
    
    # Save the standard scaler
    joblib.dump(scaler, f"{models_folder}/scaler.joblib")
    
    ###########################################################################
    ### STEP 2. FORMAT THE REPEATED CV DATASETS
    ###########################################################################
    
    # Get all of the unformatted hyperparameter tuning datasets
    raw_df_paths = glob(f"{intermediate_folder}/raw_repeated_cv_datasets/*.csv")
    
    # Define the name of the output repeated CV folder
    output_repeat_cv_folder = f"{intermediate_folder}/formatted_repeated_cv_datasets"
    
    # Iterate over the repeated CV folders
    for raw_df_path in raw_df_paths:
                
        # Load in the modeling dataframe
        cv_df = pd.read_csv(raw_df_path)
        
        # Re-code 'landcover' column into 1 for "mangrove" and 0 for "non-mangrove"
        cv_df['landcover'] = cv_df['landcover'].map({'mangrove': 1, 'nonmangrove': 0})
        
        # Define all of the features
        features = [
            'ARVI', 'ARVI_INTP', 'B', 'CMRI', 'CMRI_INTP', 'CVI', 'CVI_INTP',
            'EVI_INTP', 'EVI_RMSE', 'G', 'GLI', 'GLI_INTP', 'IPVI', 'IPVI_INTP', 
            'MCARI1', 'MCARI1_INTP', 'MCARI2', 'MCARI2_INTP', 'N', 'NDTI',
            'NDTI_INTP', 'NDVI', 'NDVI_INTP', 'NDWI', 'NDWI_INTP', 'NLI',
            'NLI_INTP', 'N_INTP', 'OSAVI', 'OSAVI_INTP', 'R', 'R_INTP', 'TDVI',
            'TDVI_INTP', 'TVI', 'TVI_INTP', 'TriVI', 'TriVI_INTP', 'WDRVI', 'WDRVI_INTP'
            ]
        
        # Initialize the StandardScaler
        scaler = StandardScaler()
        cv_df[features] = scaler.fit_transform(cv_df[features])
        
        # Save the training dataset
        output_basename = os.path.basename(raw_df_path)
        cv_df.to_csv(f"{output_repeat_cv_folder}/{output_basename}")
 