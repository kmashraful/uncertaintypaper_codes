library(sf)
library(ncf)
library(blockCV)

setwd("C:/Users/johnb/Documents/git/masters_project/stacked_generalization")

# This function generates 5 spatial folds for hyperparameter tuning
generate_tuning_dataset = function (input_spatial_df) {
  
  # Perform the randomized block to CV fold assignment
  bcv_output = cv_spatial(
    x = input_spatial_df,
    k = 5,
    size = 25000,
    hexagon = FALSE,
    selection = "random"
  )
  
  # Save the cross validation grid
  st_write(
    obj = bcv_output$blocks,
    dsn = "./intermediate_outputs/spatial_blocks_tuning.geojson",
    driver = "GeoJSON"
  )
  
  # Set the cross validation folder
  spatial_df$cv_fold_id =  bcv_output$folds_ids
  
  # Define the final output columns
  # Define the features of interest
  columns_to_keep = c(
    "ARVI", "ARVI_INTP", "B", "CMRI", "CMRI_INTP", "CVI", "CVI_INTP", 
    "EVI_INTP", "EVI_RMSE", "G", "GLI", "GLI_INTP", "IPVI", "IPVI_INTP", 
    "MCARI1", "MCARI1_INTP", "MCARI2", "MCARI2_INTP", "N", "NDTI", 
    "NDTI_INTP", "NDVI", "NDVI_INTP", "NDWI", "NDWI_INTP", "NLI", 
    "NLI_INTP", "N_INTP", "OSAVI", "OSAVI_INTP", "R", "R_INTP", 
    "TDVI", "TDVI_INTP", "TVI", "TVI_INTP", "TriVI", "TriVI_INTP", 
    "WDRVI", "WDRVI_INTP", "landcover", "cv_fold_id"
  )
  
  # Extract the feature matrix
  output_df = spatial_df[, columns_to_keep, drop = TRUE]
  
  # Save the final modeling dataset CSV
  write.csv(output_df, "./intermediate_outputs/raw_tuning_dataset.csv", row.names = F)
  
}

# This function generates 5 spatial folds for hyperparameter tuning
generate_repeat_cv_datasets = function (input_spatial_df) {
  
  # Generate 100-versions of the dataset for repeated CV
  for (i in c(1:100)) {
    
    # Perform the randomized block to CV fold assignment
    bcv_output = cv_spatial(
      x = input_spatial_df,
      k = 5,
      size = 25000,
      hexagon = FALSE,
      selection = "random"
    )
    
    # # Save the cross validation grid
    # st_write(
    #   obj = bcv_output$blocks, 
    #   dsn = "./intermediate_outputs/spatial_blocks_tuning.geojson", 
    #   driver = "GeoJSON"
    # )
    
    # Set the cross validation folder
    spatial_df$cv_fold_id =  bcv_output$folds_ids
    spatial_df$repeat_cv_id = i
    
    # Define the final output columns
    # Define the features of interest
    columns_to_keep = c(
      "ARVI", "ARVI_INTP", "B", "CMRI", "CMRI_INTP", "CVI", "CVI_INTP", 
      "EVI_INTP", "EVI_RMSE", "G", "GLI", "GLI_INTP", "IPVI", "IPVI_INTP", 
      "MCARI1", "MCARI1_INTP", "MCARI2", "MCARI2_INTP", "N", "NDTI", 
      "NDTI_INTP", "NDVI", "NDVI_INTP", "NDWI", "NDWI_INTP", "NLI", 
      "NLI_INTP", "N_INTP", "OSAVI", "OSAVI_INTP", "R", "R_INTP", 
      "TDVI", "TDVI_INTP", "TVI", "TVI_INTP", "TriVI", "TriVI_INTP", 
      "WDRVI", "WDRVI_INTP", "landcover", "cv_fold_id", "repeat_cv_id"
    )
    
    # Extract the feature matrix
    output_df = spatial_df[, columns_to_keep, drop = TRUE]
    
    # Save the final modeling dataset CSV
    output_path = paste0("./intermediate_outputs/raw_repeated_cv_datasets/cv_dataset_",i,".csv")
    write.csv(output_df, output_path, row.names = F)
    
  }
  
}

################################################################################


# Read in the modeling CSV
modeling_df = read.csv("./data/SamplePointsExport_nicfi_all.csv")
modeling_df = na.omit(modeling_df)

# Load in the previously computed correlogram
load("./outputs/correlogram.RData")

# Convert to an sf object (spatial dataframe)
spatial_df = st_as_sf(modeling_df, coords = c("longitude", "latitude"), crs = 4326)

# Reproject to EPSG:9678
spatial_df = st_transform(spatial_df, crs = 9678)

# Generate the modeling dataset used for Cross-Validation during hyper parameter tuning
generate_tuning_dataset(spatial_df)

# Generate the modeling datasets used for repeated Cross-Validation
generate_repeat_cv_datasets(spatial_df)
