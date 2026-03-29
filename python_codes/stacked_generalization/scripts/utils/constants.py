
# Define any of the seeds used for reproducibility
RANDOM_SEED = 569784 

# Define the number of optuna hyperparameter trials
N_TRIALS = 250

# Define the modeling features
FEATURES = [
    'ARVI', 'ARVI_INTP', 'B', 'CMRI', 'CMRI_INTP', 'CVI', 'CVI_INTP',
    'EVI_INTP', 'EVI_RMSE', 'G', 'GLI', 'GLI_INTP', 'IPVI', 'IPVI_INTP', 
    'MCARI1', 'MCARI1_INTP', 'MCARI2', 'MCARI2_INTP', 'N', 'NDTI',
    'NDTI_INTP', 'NDVI', 'NDVI_INTP', 'NDWI', 'NDWI_INTP', 'NLI',
    'NLI_INTP', 'N_INTP', 'OSAVI', 'OSAVI_INTP', 'R', 'R_INTP', 'TDVI',
    'TDVI_INTP', 'TVI', 'TVI_INTP', 'TriVI', 'TriVI_INTP', 'WDRVI', 'WDRVI_INTP'
    ]