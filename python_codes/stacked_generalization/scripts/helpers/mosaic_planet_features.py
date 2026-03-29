import os
from glob import glob
from osgeo import gdal

gdal.UseExceptions()

def create_vrt(input_rasters, output_vrt):
    """
    Create a VRT using GDAL from a list of input raster paths.
    Use average resampling and do not separate bands.
    """
    vrt_options = gdal.BuildVRTOptions(
        separate=False,
        resampleAlg='average',
        resolution='highest',
        srcNodata=-32768,
        VRTNodata=-32768
    )
    gdal.BuildVRT(output_vrt, input_rasters, options=vrt_options)

if __name__ == "__main__":
    
    # Define paths
    vrt_path = "C:/Users/johnb/Documents/planet_features.vrt"
    final_tif = "C:/Users/johnb/Desktop/planet_features.tif"
    raster_paths = glob("C:/Users/johnb/Desktop/Mangrove_Features/*.tif")

    # Create a VRT
    create_vrt(raster_paths, vrt_path)

    # Get resolution from the first raster (to avoid extremely large outputs)
    ds = gdal.Open(raster_paths[0])
    gt = ds.GetGeoTransform()
    x_res = abs(gt[1])
    y_res = abs(gt[5])
    ds = None

    # Warp the VRT into a single raster
    # -co TILED=YES allows for better internal tiling
    # -co BLOCKXSIZE=256 and BLOCKYSIZE=256 can help with performance
    # Deflate compression with predictor often reduces file size more effectively than default LZW
    cmd = (
        f'gdalwarp -of GTiff -r average '
        f'-srcnodata -32768 -dstnodata -32768 '
        f'-tr {x_res} {y_res} -tap '
        f'-co COMPRESS=DEFLATE -co TILED=YES '
        f'-co BLOCKXSIZE=256 -co BLOCKYSIZE=256 '
        f'-co BIGTIFF=YES '
        f'"{vrt_path}" "{final_tif}"'
    )
    os.system(cmd)
