from osgeo import gdal
import os
import numpy as np
from summary import *

# Configure GDAL
os.environ["GDAL_DRIVER_PATH"] = os.path.join(r"C:\Users\ralaya\Miniforge3\envs\mrms-index",
                                              r"Library\lib\gdalplugins")
gdal.UseExceptions()

# Load area array
pixel_areas = np.load('pixel_areas/pixel_areas_m2.npy').ravel()

# Fetch and decode data
URL = (
    'https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass1_00.00/'
    '20260526/MRMS_MultiSensor_QPE_01H_Pass1_00.00_20260526-130000.grib2.gz'
)
data = gdal.Open(f'/vsigzip//vsicurl/{URL}')
array = data.ReadAsArray().astype(np.float32).ravel()
numCols = data.RasterXSize

radar_indices = np.load('radar_indices/indices.npz')
for rid, indices in radar_indices.items():
    rain_over_radar = array[indices]

    row_indices = indices // numCols
    area_over_radar = pixel_areas[row_indices]

    # Get summaries for each radar
    area_w_rain = area_with_rain(rain_over_radar, area_over_radar)
    area = total_area(area_over_radar)

    print(
        f"Radar: {rid}\n" +
        f"area rain: {area_w_rain}\n" +
        f"total area: {area}\n" +
        f"% area raining: {area_w_rain / area}\n" +
        f"total rain: {total_rain(rain_over_radar)}\n" +
        f"mean rain: {mean_rain(rain_over_radar)}\n"
    )