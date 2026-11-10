"""Draw annotated images chosen for varied crowding AND object size.
Raw CSV boxes:
    python -m src.data.visualize_dataset --root D:\efficient-retail-detection\SKU110K_fixed --split val --out experiments/figures/val_annotations.png
Conversion check (red = CSV boxes, green = boxes decoded from exported YOLO labels):
    python -m src.data.visualize_dataset --root D:\efficient-retail-detection\SKU110K_fixed --split val --yolo-root D:\efficient-retail-detection\sku110k_subset --out experiments/figures/val_yolo_overlay.png

"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from PIL import Image
from src.data import sku110k as sk
from src.fundamentals.boxes import box_area

def rel_box_size(a: sk.ImageAnn) -> float:
    return float(np.median(np.sqrt(box_area(a.boxes)))) / max(a.width, a.height)

def pick_varied(anns: dict[str, sk.ImageAnn], n: int) -> list[str]:
    """Deterministic: half evenly spaced over box count, half over relative box size."""
    names = sorted(anns)
    by_count = sorted(names, key=lambda k: (len(anns[k]), k))
    by_size = sorted(names, key=lambda k: (rel_box_size(anns[k]), k))
    picks: list[str] = []
    for order, k in ((by_count, n - n // 2), (by_size, n // 2)):
        for i in np.linspace(0, len(order) - 1, k).round().astype(int):
            j = int(i)
            while order[j] in picks and j < len(order) - 1:
                j += 1
            if order[j] not in picks:
                picks.append(order[j])
    return picks

def load_scaled(path: Path, max_side: int = 1280) -> tuple[np.ndarray, float, float]:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w0, h0 = im.size
        s = min(1.0, max_side / max(w0, h0))
        if s < 1:
            im = im.resize((round(w0 * s), round(h0 * s)), Image.BILINEAR)
        return np.asarray(im), im.width / w0, im.height / h0
    
def add_boxes(ax, boxes: np.ndarray, color: str, lw: float) -> None:
    rects = [Rectangle((x1, y1), x2 - x1, y2 - y1) for x1, y1, x2, y2 in boxes]
    ax.add_collection(PatchCollection(rects, facecolor="none", edgecolor=color, linewidth=lw))
    
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--split", default="val", choices=sk.SPLITS)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--yolo-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("experiments/figures/annotations.png"))
    args = ap.parse_args()
    anns = sk.read_annotations(sk.csv_path(args.root, args.split))
    if args.yolo_root:
        anns = {k: v for k, v in anns.items()
                if sk.label_path(args.yolo_root, args.split, k).exists()}
        if not anns:
            raise SystemExit("no exported labels for this split")
    names = pick_varied(anns, args.n)
    cols = 4
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, name in zip(axes.flat, names):
        a = anns[name]
        img, sx, sy = load_scaled(args.root / "images" / name)
        scale = np.array([sx, sy, sx, sy])
        ax.imshow(img)
        if args.yolo_root:
            add_boxes(ax, a.boxes * scale, "red", 1.4)
            lab = sk.read_label_file(sk.label_path(args.yolo_root, args.split, name))
            add_boxes(ax, sk.from_yolo(lab, a.width, a.height) * scale, "lime", 0.5)
        else:
            add_boxes(ax, a.boxes * scale, "lime", 0.6)
        med = np.median(np.sqrt(box_area(a.boxes)))
        ax.set_title(f"{name} ({a.width}x{a.height})\n{len(a)} boxes, median side {med:.0f}px",
                     fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved {len(names)} images -> {args.out}")
    
if __name__ == "__main__":
    main()
