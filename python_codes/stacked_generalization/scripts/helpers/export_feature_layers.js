
function main () {

  // Define the export bounds 
  var aoi = ee.Geometry.Polygon([[
    [89.0153, 22.519],  [89.0153, 21.605],
    [89.940, 21.605],  [89.940, 22.519]
    ]], null, false);

  
  // Define the features to export
  var features = [
    'ARVI', 'ARVI_INTP', 'B', 'CMRI', 'CMRI_INTP', 'CVI', 'CVI_INTP',
    'EVI_INTP', 'EVI_RMSE', 'G', 'GLI', 'GLI_INTP', 'IPVI', 'IPVI_INTP',
    'MCARI1', 'MCARI1_INTP', 'MCARI2', 'MCARI2_INTP', 'N', 'NDTI',
    'NDTI_INTP', 'NDVI', 'NDVI_INTP', 'NDWI', 'NDWI_INTP', 'NLI',
    'NLI_INTP', 'N_INTP', 'OSAVI', 'OSAVI_INTP', 'R', 'R_INTP', 'TDVI',
    'TDVI_INTP', 'TVI', 'TVI_INTP', 'TriVI', 'TriVI_INTP', 'WDRVI', 'WDRVI_INTP'
    ];

  // NICFI image low NDWI 4 bands and indices
  var nicfi_image_lowndwi_ind = ee.Image("projects/ee-islamkm/assets/BD_NICFI_RAWBands_indices_w2w");
  
  // NICFI image CCDC coefs
  var nicfi_image_CCDC = ee.Image("projects/ee-islamkm/assets/CCDC_NICFI_indices_w2w");
  
  // Concatonate the two images
  var export_image = ee.Image.cat(nicfi_image_lowndwi_ind, nicfi_image_CCDC)
    .select(features)
    .unmask(-32768);
 
  // Export the data
  Export.image.toDrive({
    image: export_image.float(), 
    description: "Mangrove-Features", 
    folder: "Mangrove-Features", 
    fileNamePrefix: "mangrove_features", 
    region: aoi, 
    scale: 4.77, 
    crs: "EPSG:4326", 
    maxPixels: 1e13
  });
  
  // Map.addLayer(nicfi_image_lowndwi_ind, {}, "nicfi_image_lowndwi_ind");
  // Map.addLayer(nicfi_image_CCDC, {}, "nicfi_image_CCDC");
  Map.addLayer(export_image, {}, "export_image");
  
  return null;
  
}

main();
