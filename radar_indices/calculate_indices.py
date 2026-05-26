from shapely.geometry import Point
from shapely.ops import transform
from pyproj import Transformer, CRS
from osgeo import gdal, ogr, osr
import numpy as np
import json

radius_m = 230_000

# MRMS grid constants
x_min, y_min, x_max, y_max = -130, 20, -60, 55
x_res, y_res = 0.01, 0.01
width = 7000
height = 3500

wgs84 = CRS.from_epsg(4326)
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
srs_wkt = srs.ExportToWkt()

# Reusable raster (zeroed per radar)
mem_raster_driver = gdal.GetDriverByName("MEM")
ds = mem_raster_driver.Create("", width, height, 1, gdal.GDT_Byte)
ds.SetGeoTransform([x_min, x_res, 0, y_max, 0, -y_res])
ds.SetProjection(srs_wkt)
band = ds.GetRasterBand(1)

mem_vec_driver = ogr.GetDriverByName("Memory")

with open("radars.json") as f:
    radars = json.load(f)

results = {}

for radar in radars:
    rid = radar["id"]
    lat, lon = radar["lat"], radar["lng"]

    # AEQD centered on this radar
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
    to_aeqd = Transformer.from_crs(wgs84, aeqd, always_xy=True).transform
    to_wgs84 = Transformer.from_crs(aeqd, wgs84, always_xy=True).transform

    buffer = transform(to_wgs84, transform(to_aeqd, Point(lon, lat)).buffer(radius_m))

    # Fresh in-memory layer for this radar
    src_ds = mem_vec_driver.CreateDataSource(rid)
    layer = src_ds.CreateLayer(rid, srs=srs, geom_type=ogr.wkbPolygon)
    feat = ogr.Feature(layer.GetLayerDefn())
    feat.SetGeometry(ogr.CreateGeometryFromJson(json.dumps(buffer.__geo_interface__)))
    layer.CreateFeature(feat)

    # Reset raster and burn
    band.Fill(0)
    gdal.RasterizeLayer(ds, [1], layer, burn_values=[1])

    arr = band.ReadAsArray()
    results[rid] = np.flatnonzero(arr).astype(np.uint32)

    src_ds = None

ds = None

np.savez_compressed("indices.npz", **results)