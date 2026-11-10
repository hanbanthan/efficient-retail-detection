"""Box format conversions and area. All functions accept (..., 4) arrays."""
import numpy as np

def xyxy_to_xywh(b):
    b = np.asarray(b, dtype=float)
    out = b.copy()
    out[..., 2] = b[..., 2] - b[..., 0]
    out[..., 3] = b[..., 3] - b[..., 1]
    return out

def xywh_to_xyxy(b):
    b = np.asarray(b, dtype=float)
    out = b.copy()
    out[..., 2] = b[..., 0] + b[..., 2]
    out[..., 3] = b[..., 1] + b[..., 3]
    return out

def xyxy_to_cxcywh(b):
    b = np.asarray(b, dtype=float)
    return np.stack([(b[..., 0] + b[..., 2])/2, (b[..., 1] + b[..., 3])/2,
                     b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]], axis=-1)

def cxcywh_to_xyxy(b):
    b = np.asarray(b, dtype=float)
    return np.stack([b[..., 0] - b[..., 2]/2, b[..., 1] - b[..., 3]/2,
                     b[..., 0] + b[..., 2]/2, b[..., 1] + b[..., 3]/2], axis=-1)

def box_area(b):
    b = np.asarray(b, dtype=float)
    w = np.clip(b[..., 2] - b[..., 0], 0, None)
    h = np.clip(b[..., 3] - b[..., 1], 0, None)
    return w * h

