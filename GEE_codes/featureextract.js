// Planet data: Bands and indices


// Load your study area (AOI)
var aoi = ee.FeatureCollection('projects/ee-islamkm/assets/sundarban_bd_aoi_w2w');


// #####################################

// Set map center for visualization
Map.setCenter(89.6307, 22.0602, 10);

// Function to apply cloud and shadow filtering to NICFI images
var filterClouds = function(img) {
  // Extract the blue band and near-infrared band
  var blueBand = img.select('B');
  var nearBand = img.select('N');
  
  // Apply the cloud mask using a threshold value for blue reflectance
  var cloudMask = blueBand.lte(900); // Adjust threshold as needed
  
  // Apply the shadow mask using a threshold value for near-infrared reflectance
  var shadowMask = nearBand.lte(3500); // Adjust threshold as needed
  
  // Combine cloud and shadow masks
  var combinedMask = cloudMask.and(shadowMask);
  
  // Apply the combined mask to the image
  var maskedImage = img.updateMask(combinedMask);
  
  // Set the system time property for tracking
  return maskedImage.set('system:time_start', img.get('system:time_start'));
};


// Load the NICFI basemaps for the specified date range
var nicfi = ee.ImageCollection('projects/planet-nicfi/assets/basemaps/asia')
                .filter(ee.Filter.date('2020-11-01','2021-03-31')).map(filterClouds).map(function(image) {
                            return image.clip(aoi);
                            });
                            


// Load the spectral indices utilities

// *CREDIT*: Montero, D., Aybar, C., Mahecha, M. D., Wieneke, S. (2022). spectral: Awesome Spectral Indices deployed
// via the Google Earth Engine JavaScript API. The International Archives of the Photogrammetry, Remote Sensing
// and Spatial Information Sciences, Volume XLVIII-4/W1-2022. Free and Open Source Software for Geospatial
// (FOSS4G) 2022 Academic Track, 22-28 August 2022, Florence, Italy. 
// doi: 10.5194/isprs-archives-XLVIII-4-W1-2022-301-2022

var spectral = require("users/dmlmont/spectral:spectral");

// Selected spectral indices to compute
var selected_indices = ["ARVI", "AVI", "CVI", "ExG", "EVI", "FCVI", "GARI", "GBNDVI", "GLI", "GNDVI", 
                        "GOSAVI", "IPVI", "MCARI1", "MCARI2", "MSAVI", "MSR", "NDTI", "NDVI", "NDWI", 
                        "NLI", "OSAVI", "SI", "TDVI", "TGI", "TriVI", "TVI", "WDRVI"];
                                    
// DATASET TO USE: 
var dataset = 'projects/planet-nicfi/assets/basemaps/asia';

// Function to add spectral indices to images in the collection
function addIndices(img) {
  var date = img.get('system:time_start');
  var parameters = {
    "B": img.select("B"),
    "G": img.select("G"),
    "R": img.select("R"),
    "N": img.select("N"),
    "g": 2.5,
    "C1": 6,
    "C2": 7.5,
    "L": 1,
    "alpha": 0.1,
    "beta": 0.05,
    "c": 1.0,
    "cexp": 1.16,
    "epsilon": 1,
    "fdelta": 0.581,
    "gamma": 1.0,
    "k": 0.0,
    "nexp": 2.0,
    "omega": 2.0,
    "p": 2.0,
    "sigma": 0.5,
    "sla": 1.0,
    "slb": 0.0,
    "kernel":'RBF',
    "kNN": 1.0,
    "kNR": spectral.computeKernel(img,"RBF",{
        "a": img.select("N"),
        "b": img.select("R"),
        "sigma": 0.5
    })
        
  };
  img = spectral.scale(img, dataset);
  return spectral.computeIndex(img, selected_indices, parameters);
}

// Apply the function to add indices to the image collection
var nicfi_with_severalindices = nicfi.map(addIndices);

// Function to get CMRI
var mangrove_index = function(img) {
  // Clip the image to the AOI
  var clipped = img.clip(aoi);
  
  
  // Calculate CMRI (NDVI - NDWI)
  var cmri = (clipped.select('NDVI')).subtract(clipped.select('NDWI')).rename('CMRI');
  
  // Add the new bands (NDVI, NDWI, CMRI) to the image and retain 'system:time_start'
  return clipped.addBands([cmri])
                .set('system:time_start', img.get('system:time_start'));
};

// Map the function over the NICFI ImageCollection
var nicfi_with_bands_indices = nicfi_with_severalindices.map(mangrove_index);

// LOWEST NDVI VALUE COMPOSITE
// Example function to add an inverted NDWI band as a "quality" band for the mosaic
var addInvertedNdviQuality = function(img) {
  var ndvi = img.select('NDWI');  // Select the NDWI band
  var ndviInv = ndvi.multiply(-1).rename('quality');  // Invert the NDWI values
  return img.addBands(ndviInv);  // Add the inverted NDVI as the "quality" band
};

nicfi_with_bands_indices = nicfi_with_bands_indices.map(addInvertedNdviQuality);

// Create the quality mosaic based on the inverted NDVI values
var qualityMosaic_low_ndwi = nicfi_with_bands_indices.qualityMosaic('quality')
.select("B", "G", "R", "N", "CMRI", "EVI", "OSAVI", 
                    "NDVI", "NDWI", "ARVI", "CVI", "GLI", "IPVI", "MCARI1", "MCARI2", "NDTI", "NLI", "TDVI", "TGI", "TriVI", "TVI", "WDRVI");


print("Quality mosaic info: ", qualityMosaic_low_ndwi)

// Export NICFI results to an Earth Engine asset
Export.image.toAsset({
  image: qualityMosaic_low_ndwi,
  description: 'NICFI_RAWBands_indices',
  assetId: 'BD_NICFI_RAWBands_indices_w2w',
  region: aoi.geometry(),
  scale: 4.77, // NICFI Planet data is 4.77m resolution
  maxPixels: 1e9
});
