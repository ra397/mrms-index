import numpy as np
from osgeo import gdal, osr

rid = "KBYX"

xmin, ymin, xmax, ymax = -130.0, 20.0, -60.0, 55.0
res = 0.01
width, height = 7000, 3500

data = np.load("indices.npz")
indices = data[rid].astype(np.uint32)

arr = np.zeros(height * width, dtype=np.uint8)

arr[indices] = 1
arr = arr.reshape((height, width))

driver = gdal.GetDriverByName("GTiff")
ds = driver.Create(
    f"{rid}.tif",
    width,
    height,
    1,
    gdal.GDT_Byte,
    options=["COMPRESS=LZW", "TILED=YES"],
)

geotransform = (xmin, res, 0.0, ymax, 0.0, -res)
ds.SetGeoTransform(geotransform)

srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
ds.SetProjection(srs.ExportToWkt())

band = ds.GetRasterBand(1)
band.WriteArray(arr)
band.SetNoDataValue(0)
band.FlushCache()

ds = None