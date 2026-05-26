import numpy as np
from pyproj import Geod

xmin, ymin, xmax, ymax = -130.0, 20.0, -60.0, 55.0
xres, yres = 0.01, 0.01
ncols, nrows = 7000, 3500

row_idx = np.arange(nrows)

lat_top = ymax - row_idx * yres
lat_bot = ymax - (row_idx + 1) * yres

geod = Geod(ellps="WGS84")

lon_left = 0.0
lon_right = lon_left + xres

areas_m2 = np.empty(nrows, dtype=np.float64)
for i in range(nrows):
    lons = [lon_left, lon_right, lon_right, lon_left]
    lats = [lat_bot[i], lat_bot[i], lat_top[i], lat_top[i]]
    area, _ = geod.polygon_area_perimeter(lons, lats)
    areas_m2[i] = abs(area)

np.save("pixel_areas_m2.npy", areas_m2)