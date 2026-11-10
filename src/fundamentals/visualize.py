import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_boxes(ax, image, boxes, scores=None, color="lime"):
    ax.imshow(image)
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes, dtype=float)):
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor=color, linewidth=2))
        if scores is not None:
            ax.text(x1, y1 - 1, f"{scores[i]:.2f}", color=color, fontsize=8)
    ax.axis("off")