import numpy as np

def area_with_rain(values, areas, threshold = 0.0):
    return np.sum(areas[values > threshold])

def total_area(areas):
    return np.sum(areas)

def total_rain(values):
    return np.sum(values > 0.0)

def mean_rain(rain, threshold = 0.0):
    if not np.any(rain > threshold):
        return np.nan
    return np.mean(rain[rain > threshold])