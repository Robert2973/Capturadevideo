import cv2
import numpy as np
import utils

# --- recolors (channel mixing) ---
def recolorRC(src, dst):
    b, g, r = cv2.split(src)
    cv2.addWeighted(b, 0.5, g, 0.5, 0, b)
    cv2.merge((b, b, r), dst)

def recolorRGV(src, dst):
    b, g, r = cv2.split(src)
    cv2.min(b, g, b)
    cv2.min(b, r, b)
    cv2.merge((b, g, r), dst)

def recolorCMV(src, dst):
    b, g, r = cv2.split(src)
    cv2.max(b, g, b)
    cv2.max(b, r, b)
    cv2.merge((b, g, r), dst)

# --- curves & helpers (classes) ---
class VFuncFilter(object):
    def __init__(self, vFunc=None, dtype=np.uint8):
        length = np.iinfo(dtype).max + 1
        self._vLookupArray = utils.createLookupArray(vFunc, length)

    def apply(self, src, dst):
        srcFlat = utils.flatView(src)
        dstFlat = utils.flatView(dst)
        utils.applyLookupArray(self._vLookupArray, srcFlat, dstFlat)

class VCurveFilter(VFuncFilter):
    def __init__(self, vPoints=None, dtype=np.uint8):
        super().__init__(utils.createCurveFunc(vPoints), dtype)

class BGRFuncFilter(object):
    def __init__(self, vFunc=None, bFunc=None, gFunc=None, rFunc=None, dtype=np.uint8):
        length = np.iinfo(dtype).max + 1
        def comp(a, b): return (lambda x: a(b(x))) if (a is not None and b is not None) else (a if b is None else b)
        # compose v and per-channel: v then channel
        self._bLookupArray = utils.createLookupArray(utils.createCompositeFunc(bFunc, vFunc) if hasattr(utils, 'createCompositeFunc') else None, length)
        self._gLookupArray = utils.createLookupArray(utils.createCompositeFunc(gFunc, vFunc) if hasattr(utils, 'createCompositeFunc') else None, length)
        self._rLookupArray = utils.createLookupArray(utils.createCompositeFunc(rFunc, vFunc) if hasattr(utils, 'createCompositeFunc') else None, length)

    def apply(self, src, dst):
        b, g, r = cv2.split(src)
        if self._bLookupArray is not None: utils.applyLookupArray(self._bLookupArray, b, b)
        if self._gLookupArray is not None: utils.applyLookupArray(self._gLookupArray, g, g)
        if self._rLookupArray is not None: utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([b, g, r], dst)

# fallback for createCompositeFunc if not in utils (book had it)
def createCompositeFunc(func0, func1):
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    return lambda x: func0(func1(x))

# monkey-patch into utils if missing
if not hasattr(utils, 'createCompositeFunc'):
    utils.createCompositeFunc = createCompositeFunc

class BGRCurveFilter(BGRFuncFilter):
    def __init__(self, vPoints=None, bPoints=None, gPoints=None, rPoints=None, dtype=np.uint8):
        super().__init__(utils.createCurveFunc(vPoints), utils.createCurveFunc(bPoints),
                         utils.createCurveFunc(gPoints), utils.createCurveFunc(rPoints), dtype)

# --- film emulation filters ---
class BGRPortraCurveFilter(BGRCurveFilter):
    def __init__(self, dtype=np.uint8):
        super().__init__(vPoints=[(0,0),(23,20),(157,173),(255,255)],
                         bPoints=[(0,0),(41,46),(231,228),(255,255)],
                         gPoints=[(0,0),(52,47),(189,196),(255,255)],
                         rPoints=[(0,0),(69,69),(213,218),(255,255)],
                         dtype=dtype)

class BGRProviaCurveFilter(BGRCurveFilter):
    def __init__(self, dtype=np.uint8):
        super().__init__(bPoints=[(0,0),(35,25),(205,227),(255,255)],
                         gPoints=[(0,0),(27,21),(196,207),(255,255)],
                         rPoints=[(0,0),(59,54),(202,210),(255,255)],
                         dtype=dtype)

class BGRVelviaCurveFilter(BGRCurveFilter):
    def __init__(self, dtype=np.uint8):
        super().__init__(vPoints=[(0,0),(128,118),(221,215),(255,255)],
                         bPoints=[(0,0),(25,21),(122,153),(165,206),(255,255)],
                         gPoints=[(0,0),(25,21),(95,102),(181,208),(255,255)],
                         rPoints=[(0,0),(41,28),(183,209),(255,255)],
                         dtype=dtype)

class BGRCrossProcessCurveFilter(BGRCurveFilter):
    def __init__(self, dtype=np.uint8):
        super().__init__(bPoints=[(0,20),(255,235)],
                         gPoints=[(0,0),(56,39),(208,226),(255,255)],
                         rPoints=[(0,0),(56,22),(211,255),(255,255)],
                         dtype=dtype)

# --- stroke edges (comic) ---
def strokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    if blurKsize >= 3:
        blurred = cv2.medianBlur(src, blurKsize)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(gray, cv2.CV_8U, gray, ksize=edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - gray)
    channels = cv2.split(src)
    for ch in channels:
        ch[:] = ch * normalizedInverseAlpha
    cv2.merge(channels, dst)

# --- convolution filters ---
class VConvolutionFilter(object):
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        cv2.filter2D(src, -1, self._kernel, dst)

class SharpenFilter(VConvolutionFilter):
    def __init__(self):
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        super().__init__(kernel)

class FindEdgesFilter(VConvolutionFilter):
    def __init__(self):
        kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        super().__init__(kernel)

class BlurFilter(VConvolutionFilter):
    def __init__(self):
        kernel = np.ones((5,5), dtype=np.float32) / 25.0
        super().__init__(kernel)

class EmbossFilter(VConvolutionFilter):
    def __init__(self):
        kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
        super().__init__(kernel)
