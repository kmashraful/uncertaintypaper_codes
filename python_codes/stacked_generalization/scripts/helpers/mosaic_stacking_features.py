import os
import subprocess
from glob import glob
from osgeo import gdal

gdal.UseExceptions()


def create_stacking_vrt(input_rasters, output_vrt):
    vrt_options = gdal.BuildVRTOptions(
        separate=True,
        resampleAlg='average',
        resolution='highest',
        srcNodata=-32768,
        VRTNodata=-32768
    )
    gdal.BuildVRT(output_vrt, input_rasters, options=vrt_options)


if __name__ == "__main__":
    planet_geotiff_path = "C:/Users/johnb/Desktop/planet_features.tif"
    stacking_vrt_path = "C:/Users/johnb/Desktop/sundarbans_cover_maps/vrts/stacking.vrt"
    output_geotiff = "C:/Users/johnb/Desktop/stacking_features.tif"
    intermediate_vrt_folder = "C:/Users/johnb/Desktop/sundarbans_cover_maps/vrts/intermediate_vrts"
    
    if not os.path.exists(intermediate_vrt_folder):
        os.makedirs(intermediate_vrt_folder)

    baselearner_raster_folder = "C:/Users/johnb/Desktop/sundarbans_cover_maps/baselearner_outputs"
    baselearner_rf_path = os.path.join(baselearner_raster_folder, "baselearner_rf_prediction.tif")
    baselearner_svc_path = os.path.join(baselearner_raster_folder, "baselearner_svc_prediction.tif")
    baselearner_knn_path = os.path.join(baselearner_raster_folder, "baselearner_knn_prediction.tif")
    baselearner_logreg_path = os.path.join(baselearner_raster_folder, "baselearner_logreg_prediction.tif")
    baselearner_xgb_path = os.path.join(baselearner_raster_folder, "baselearner_xgb_prediction.tif")

    baselearner_paths = [
        baselearner_rf_path,
        baselearner_svc_path,
        baselearner_knn_path,
        baselearner_logreg_path,
        baselearner_xgb_path
    ]

    # Create single-band VRTs for each band in the planet raster
    planet_ds = gdal.Open(planet_geotiff_path)
    band_count = planet_ds.RasterCount
    planet_ds = None

    planet_vrt_list = []
    for i in range(1, band_count + 1):
        single_band_vrt = os.path.join(intermediate_vrt_folder, f"planet_band_{i}.vrt")
        cmd = [
            "gdal_translate",
            "-b", str(i),
            "-of", "VRT",
            planet_geotiff_path,
            single_band_vrt
        ]
        subprocess.run(cmd, check=True)
        planet_vrt_list.append(single_band_vrt)

    all_vrt_paths = baselearner_paths + planet_vrt_list
    create_stacking_vrt(all_vrt_paths, stacking_vrt_path)

    ds = gdal.Open(planet_geotiff_path)
    gt = ds.GetGeoTransform()
    x_res = abs(gt[1])
    y_res = abs(gt[5])
    ds = None

    cmd = (
        f'gdalwarp -of GTiff -r average '
        f'-srcnodata -32768 -dstnodata -32768 '
        f'-tr {x_res} {y_res} -tap '
        f'-co COMPRESS=DEFLATE -co TILED=YES '
        f'-co BLOCKXSIZE=256 -co BLOCKYSIZE=256 '
        f'-co BIGTIFF=YES '
        f'"{stacking_vrt_path}" "{output_geotiff}"'
    )
    os.system(cmd)
