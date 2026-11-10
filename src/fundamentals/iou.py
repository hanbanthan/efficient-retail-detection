import numpy as np
from src.fundamentals.boxes import box_area

def iou_matrix(a, b):
    """Pairwise IoU between xyxy boxes a (N, 4) and b (M, 4) -> (N, M)"""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    safe = np.where(union > 0, union, 1.0)
    return np.where(union > 0, inter/safe, 0.0)

def iou(box_a, box_b) -> float:
    return float(iou_matrix(box_a, box_b)[0, 0])