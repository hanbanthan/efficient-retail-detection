import numpy as np
from src.fundamentals.iou import iou_matrix

def nms(boxes, scores, iou_threshold=0.5):
    """Greedy NMS. Returns kept indices, highest score first.
    A box is suppressed if its IoU with an already-kept box is > iou_threshold."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float)
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores, kind='stable')
    ious = iou_matrix(boxes, boxes)
    suppressed = np.zeros(len(boxes), dtype=bool)
    keep = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        suppressed |= ious[i] > iou_threshold
    return keep