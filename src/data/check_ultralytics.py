"""Load the exported subset with Ultralytics' own dataset class and draw the letterboxed samples.

    python -m src.data.check_ultralytics --data D:/efficient-retail-detection/sku110k_subset/data.yaml --split val --out experiments/figures/ultralytics_val.png
Written for the ultralytics 8.3.x API (YOLODataset, check_det_dataset).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics.data import YOLODataset
from ultralytics.data.utils import check_det_dataset, img2label_paths
from ultralytics.utils import DEFAULT_CFG

from src.data import sku110k as sk
from src.data.visualize_dataset import add_boxes
from src.fundamentals.boxes import cxcywh_to_xyxy


def load_dataset(data_yaml: str, split: str = "val", imgsz: int = 640) -> YOLODataset:
    data = check_det_dataset(data_yaml)
    assert data["nc"] == 1, f"expected one class, got {data['names']}"
    return YOLODataset(img_path=data[split], imgsz=imgsz, augment=False, hyp=DEFAULT_CFG,
                       data=data, task="detect", batch_size=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("experiments/figures/ultralytics_check.png"))
    args = ap.parse_args()

    ds = load_dataset(args.data, args.split, args.imgsz)
    n_mismatch = 0
    for lab in ds.labels:
        n_file = len(sk.read_label_file(Path(img2label_paths([lab["im_file"]])[0])))
        n_mismatch += int(n_file != len(lab["cls"]))
    max_boxes = max(len(l["cls"]) for l in ds.labels)
    print(f"{len(ds)} images loaded; label-count mismatches vs .txt: {n_mismatch}; "
          f"max boxes/image: {max_boxes} (Ultralytics default max_det=300)")

    cols = min(args.n, 4)
    fig, axes = plt.subplots(math.ceil(args.n / cols), cols, figsize=(5 * cols, 5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, range(min(args.n, len(ds)))):
        item = ds[i]
        img = item["img"].permute(1, 2, 0).numpy()          # CHW RGB uint8, letterboxed
        h, w = img.shape[:2]
        boxes = cxcywh_to_xyxy(item["bboxes"].numpy() * np.array([w, h, w, h]))
        ax.imshow(img)
        add_boxes(ax, boxes, "lime", 0.5)
        ax.set_title(f"{Path(item['im_file']).name}: {len(boxes)} boxes @ {w}x{h}", fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()