import numpy as np
import matplotlib.pyplot as plt
from src.fundamentals.nms import nms
from src.fundamentals.visualize import draw_boxes

# Two real, adjacent products (P1, P2) plus a duplicate detection of P1 (D).
boxes = np.array([[0, 0, 10, 10], [0.5, 0.5, 10.5, 10.5], [5, 0, 15, 10]], float)
scores = np.array([0.90, 0.85, 0.80])
img = np.full((14, 20, 3), 0.9)

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, thr, title in zip(axes, (0.9, 0.5, 0.3),
                          ("thr 0.9: duplicate kept", "thr 0.5: correct", "thr 0.3: product missed")):
    keep = nms(boxes, scores, thr)
    draw_boxes(ax, img, boxes[keep], scores[keep])
    ax.set_title(f"{title} (count={len(keep)}, true=2)")
fig.savefig("experiments/nms_demo.png", dpi=150, bbox_inches="tight")