"""Inspect a run: learning curves, fit diagnosis, reload best.pt, SUBSET-val metrics, examples.
    python -m src.training.inspect_run --run runs/sku110k/yolo11n_sub400_640_s0 \
        --data D:/efficient-retail-detection/sku110k_subset/data.yaml
Outputs in <run>/inspect/: curves.png, qualitative.png, metrics_subset_val.json, val/ (Ultralytics plots)
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO
from src.data import sku110k as sk
from src.data.visualize_dataset import add_boxes, load_scaled
from src.training.common import environment_info, resolve_data_yaml

LOSSES = ("box_loss", "cls_loss", "dfl_loss")
MAP50, MAP = "metrics/mAP50(B)", "metrics/mAP50-95(B)"

def read_results(path: str | Path) -> dict[str, np.ndarray]:
    """results.csv -> {column: array}. Strips the padded column names some versions write."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        k.strip(): np.array([float(r[k]) for r in rows]) for k in rows[0]
    }

def plot_curves(res: dict, out: Path) -> None:
    ep = res["epoch"]
    fig, axes = plt.subplot(1, 4, figsize=(18, 4))
    for ax, loss in zip(axes, LOSSES):
        ax.plot(ep, res[f"train/{loss}"], label="train")
        ax.plot(ep, res[f"val/{loss}"], label="val")
        ax.set_title(loss);
        ax.set_xlabel("epoch")
        ax.legend()
        
    axes[3].plot(ep, res[MAP50], label="mAP50")
    axes[3].plot(ep, res[MAP], label="mAP50-95")
    axes[3].set_title("val metrics (subset)")
    axes[3].set_xlabel("epoch");
    axes[3].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130);
    plt.close(fig)
    
def diagnose(res: dict, window: int = 5) -> list[str]:
    """Heuristic HINTS, not verdicts: confirm them with the curves and the example predictions.
    Train losses are computed on augmented (mosaic) batches, so train > val loss is normal.
    Watch the TRENDS, not the absolute gap."""
    ep, m = res["epoch"], res[MAP]
    val = sum(res[f"val/{l}"] for l in LOSSES)
    tr = sum(res[f"train/{l}"] for l in LOSSES)
    best, vmin = int(np.argmax(m)), int(np.argmin(val))
    notes = [f"best mAP50-95 = {m[best]:.4f} at epoch {int(ep[best])} / {int(ep[-1])}"]
    if len(ep) > window:
        if best >= len(m) - window and m[-1] > m[-window -1]:
            notes.append("mAP still rising at the end -> likely UNDER-trained: more epochs "
                         "(or a larger imgsz) should help.")
        if vmin < len(val) - window and val[-1] > 1.05 * val[vmin] and tr[-1] < tr[vmin]:
            notes.append(f"val loss up >5% since epoch {int(ep[vmin])} while train loss kept "
                         "falling -> OVERFITTING signal: more images / stronger augmentation / "
                         "fewer epochs. (best.pt already holds the best-fitness epoch.)")
    if m.max() < 0.10:
        notes.append("mAP < 0.10 -> check the label overlay, max_det and imgsz before tuning.")
    return notes

def pick_by_gt_count(label_dir: Path, k: int) -> list[Path]:
    """Validation images spread from the fewest to the most GT boxes (not cherry-picked)."""
    files = sorted(label_dir.glob("*.txt"), key=lambda p: (len(sk.read_label_file(p)), p.name))
    idx = np.unique(np.linspace(0, len(files) - 1, k).round().astype(int))
    return [files[i] for i in idx]

def qualitative(model, data_dir: Path, k: int, imgsz: int, max_det: int, conf: float, out: Path) -> None:
    labels = pick_by_gt_count(data_dir / "labels" / "val", k)
    cols = min(3, len(labels))
    rows = int(np.ceil(len(labels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, lbl in zip(axes.flat, labels):
        img_path = data_dir / "images" / "val" / f"{lbl.stem}.jpg"
        res = model.predict(str(img_path), imgsz=imgsz, max_det=max_det, conf=conf, verbose=False)[0]
        h, w = res.orig_shape
        img, sx, sy = load_scaled(img_path)
        s = np.array([sx, sy, sx, sy])
        gt = sk.from_yolo(sk.read_label_file(lbl), w, h)
        ax.imshow(img)
        add_boxes(ax, gt * s, "red", 1.2) # ground truth
        add_boxes(ax, res.boxes.xyxy.cpu().numpy()*s, "lime", 0.6) # prediction
        ax.set_title(f"{img_path.name} GT={len(gt)} pred={len(res.boxes)} (conf>={conf})", fontsize=9)
    
    fig.suptitle("red = ground truth, green = prediction (subset val)")
    fig.tight_layout();
    fig.savefig(out, dpi=120);
    plt.close()
    
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-qual", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.25)
    # display threshold only; mAP uses 0.001
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    
    out = a.run / "inspect"
    out.mkdir(exist_ok=True)
    run_args = yaml.safe_load((a.run / "args.yaml").read_text())
    imgsz, max_det = run_args["imgsz"], run_args["max_det"] # evaluate exactly as trained
    
    res = read_results(a.run / "results.csv")
    plot_curves(res, out / "curves.png")
    notes = diagnose(res)
    print("\n".join(notes))
    
    data_yaml = resolve_data_yaml(a.data)
    best = a.run / "weights" / "best.pt"
    model = YOLO(str(best)) # <- the "reloads" check
    m = model.val(data=str(data_yaml), split="val", imgsz=imgsz, max_det=max_det, batch=a.batch, conf=0.001, iou=0.7, plots=True, project=str(out), name="val", exist_ok=True)
    
    manifest_path = data_yaml.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    report = {
        "scope": "SUBSET",                                     # never let this number travel unlabelled
        "split": "val",
        "n_val_images": manifest.get("splits", {}).get("val", {}).get("n_images"),
        "subset_seed": manifest.get("seed"),
        "subset_manifest_sha256": sk.sha256(manifest_path) if manifest_path.exists() else None,
        "checkpoint": str(best), "checkpoint_sha256": sk.sha256(best),
        "imgsz": imgsz, "max_det": max_det, "conf": 0.001, "iou": 0.7,
        "metrics": {"mAP50-95": float(m.box.map), "mAP50": float(m.box.map50),
                    "precision": float(m.box.mp), "recall": float(m.box.mr)},
        "fit_notes": notes,
        "note": "Subset validation metrics; NOT comparable to full-split published SKU-110K results.",
        "env": environment_info(),
    }
    (out / "metrics_subset_val.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["metrics"], indent=2))
    
    qualitative(model, data_yaml.parent, a.n_qual, imgsz, max_det, a.conf, out / "qualitative.png")
    print(f"-> {out}")
    
if __name__ == "__main__":
    main()