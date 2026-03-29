library(sf)
library(ncf)

setwd("C:/Users/johnb/Documents/git/masters_project/stacked_generalization")

# Read in the modeling CSV
dataset_path = "./data/SamplePointsExport_nicfi_all.csv"
modeling_df = read.csv(dataset_path)

# Remove rows with NA values
modeling_df = na.omit(modeling_df)

# Convert to an sf object (spatial dataframe)
spatial_df = st_as_sf(modeling_df, coords = c("longitude", "latitude"), crs = 4326)

# Reproject to EPSG:9678
spatial_df = st_transform(spatial_df, crs = 9678)

# Define the features of interest
features = c(
  "ARVI", "ARVI_INTP", "B", "CMRI", "CMRI_INTP", "CVI", "CVI_INTP", 
  "EVI_INTP", "EVI_RMSE", "G", "GLI", "GLI_INTP", "IPVI", "IPVI_INTP", 
  "MCARI1", "MCARI1_INTP", "MCARI2", "MCARI2_INTP", "N", "NDTI", 
  "NDTI_INTP", "NDVI", "NDVI_INTP", "NDWI", "NDWI_INTP", "NLI", 
  "NLI_INTP", "N_INTP", "OSAVI", "OSAVI_INTP", "R", "R_INTP", 
  "TDVI", "TDVI_INTP", "TVI", "TVI_INTP", "TriVI", "TriVI_INTP", 
  "WDRVI", "WDRVI_INTP"
)

# Extract the feature matrix
feat_matrix = spatial_df[, features, drop = TRUE]

# Standardize the feature matrix
feat_matrix_scaled = scale(feat_matrix)

# Extract original coordinates
coords = st_coordinates(spatial_df)

# Fit the multivariate correlogram
correlogram = correlog.nc(
  x = coords[, 1],      
  y = coords[, 2],         
  z = feat_matrix_scaled, 
  increment = 250,    
  resamp = 1000,
  latlon = FALSE    
)

# Save the plot to a PNG file
png("./outputs/correlogram_plot.png", width = 1500, height = 1500, res = 300)

# Extract the p-values from the correlogram
p_values = correlogram$p

# Determine marker styles
pch_values = ifelse(p_values < 0.05, 16, 1) # 16 = solid circle, 1 = hollow circle

# Plot the correlogram with x-axis limited to 25 km
par(pty = 's')
plot(
  correlogram$mean.of.class / 1000,
  correlogram$correlation,
  xlab = "Distance (km)",
  ylab = "Mantel correlation",
  xlim = c(0, 50),
  ylim = c(-0.50, 0.50),
  axes = FALSE, # Suppress default axes
  pch = pch_values # Apply marker styles based on p-values
)
axis(1)
axis(2, at = seq(-0.5, 0.5, by = 0.1), labels = seq(-0.5, 0.5, by = 0.1))

# Add a horizontal black line at y = 0
abline(h = 0, col = "black", lwd = 2)

# Add a vertical red line at correlogram$x.intercept
abline(v = 25, col = "red", lwd = 2)

# Close the graphics device to save the file
dev.off()

# Save the correlogram object to a file
save(correlogram, file = "./outputs/correlogram.RData")

