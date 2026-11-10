"""
Phase 4, step 1: run the UNMODIFIED COCO-pretrained YOLO11n on subset VALIDATION images.
    python -m src.training.pretrained_inference --data D:/efficient-retail-detection/sku110k_subset/data.yaml \
        --n 8 --out experiments/pretrained_yolo11n
        
What to expect: COCO has no "product" class. The model finds a handful of boxes labelled
bottle / cup / book / ... instead of ~150 products. That gap is the reason to fine-tune.
Outputs: <out>/images/*.jpg (annotated) and <out>/predictions.json.
Never point this at test images.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO
from src.data import sku110k as sk
from src.training.common import environment_info, resolve_data_yaml, val_image_paths

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weight", default="yolo11n.pt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    # Ultralytics' default display threshold
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("experiments/pretrained_yolo11n"))
    args = ap.parse_args()
    
    data_yaml = resolve_data_yaml(args.data)
    images = val_image_paths(data_yaml)[: args.n]
    # first n of a sorted list -> deterministic
    model = YOLO(args.weights)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    
    records = []
    # stream=True yields one Results object at a time instead of hodling them all in memory
    for res in model.predict(source=[str(p) for p in images], imgsz=args.imgsz, conf=args.conf, max_det=args.max_det, stream=True, verbose=False):
        name = Path(res.path).name
        gt = len(sk.read_label_file(sk.label_path(data_yaml.parent, "val", name)))
        cls = res.boxes.cls.int().tolist()
        # res.plot() returns a a BGR array (Open CV convention); flip to RGB before saving with PIL
        Image.fromarray(res.plot(line_width=1)[...,::-1]).save(args.out / "images" / name)
        records.append({
            "image": name,
            "gt_boxes": gt,
            "pred_boxes": len(cls),
            "pred_classes": dict(Counter(res.names[c] for c in cls)),
            "boxes_xyxy": np.round(res.boxes.xyxy.cpu().numpy(), 1).tolist(),
            "scores": np.round(res.boxes.conf.cpu().numpy(), 3).tolist(),
        })
        top = Counter(res.names[c] for c in cls).most_common(3)
        print(f"{name:>14}: GT={gt:4d}  pred={len(cls):4d}  top classes={top}")
        
    report = {"weights": args.weights, "imgsz": args.imgsz, "conf": args.conf,
              "max_det": args.max_det, "data": str(data_yaml), "env": environment_info(),
              "images": records}
    (args.out / "predictions.json").write_text(json.dumps(report, indent=2))
    print(f"-> {args.out}")

if __name__ == "__main__":
    main()
    
