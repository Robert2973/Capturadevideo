import numpy as np
import scipy.interpolate

def createCurveFunc(points):
    """Return interpolation function from control points or None."""
    if points is None:
        return None
    numPoints = len(points)
    if numPoints < 2:
        return None
    xs, ys = zip(*points)
    kind = 'cubic' if numPoints >= 4 else 'linear'
    return scipy.interpolate.interp1d(xs, ys, kind=kind, bounds_error=False, fill_value="extrapolate")

def createLookupArray(func, length=256):
    if func is None:
        return None
    lookup = np.empty(length, dtype=np.float32)
    for i in range(length):
        v = float(func(i))
        if v < 0: v = 0
        if v > length - 1: v = length - 1
        lookup[i] = v
    return lookup.astype(np.uint8)

def applyLookupArray(lookupArray, src, dst):
    if lookupArray is None:
        return
    # numpy advanced indexing handles broadcasting and dtypes
    dst[:] = lookupArray[src]

def flatView(array):
    flat = array.view()
    flat.shape = array.size
    return flat
