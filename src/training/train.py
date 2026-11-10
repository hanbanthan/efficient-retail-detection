"""Train / smoke-test / resume YOLO11n on the SKU-110K subset.
Smoke test : python -m src.training.train --config configs/yolo11n_subset.yaml --smoke
Full run   : python -m src.training.train --config configs/yolo11n_subset.yaml
Resume     : python -m src.training.train --config configs/yolo11n_subset.yaml --resume
Colab      : add --data /content/sku110k_subset/data.yaml --project /content/drive/MyDrive/erd/runs
Overrides  : --set batch=8 epochs=30   (any YOLO.train argument)
Run folder (Ultralytics + ours):
  weights/last.pt  weights/best.pt   args.yaml  results.csv  (*.png plots)   <- Ultralytics
  run_meta.json  train_config.yaml  pip_freeze.txt                           <- this script
"""

from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from ultralytics import YOLO
from src.data import sku110k as sk
from src.training.common import (environment_info, load_config, parse_overrides,
                                 pip_freeze, resolve_data_yaml, val_image_paths)

def build_train_args(cfg: dict, smoke: bool = False, overrides: dict | None = None) -> dict:
    """Merge config sections into kwargs for YOLO.train(). Pure function -> unit-testable.
    Priority: train section < smoke section < CLI overrides < data/project/name."""
    args = dict(cfg["train"])
    name = cfg["name"] 
    if smoke: 
        args.update(cfg.get("smoke", {}))
        name += "_smoke" #never mix smoke checkpoints with reals one
    
    
    args.update(overrides or {})
    args.update(data=str(cfg["data"]), project=str(Path(cfg["project"]).resolve()), name=name, exist_ok=False)
    
    return args

def write_metadata(trainer, cfg: dict, config_path: str | Path | None) -> None:
    """Ultralytics callback (on_train_start). Runs on the first start AND on every resume.
    Ultralytics already saves args.yaml/results.csv; we add provenance it doesn't record."""
    save_dir = Path(trainer.save_dir)
    meta_path = save_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"sessions": []}
    
    if "config" not in meta:
        # first session only
        manifest = Path(trainer.args.data).parent / "manifest.json"
        weights = Path(cfg["model"])
        meta.update(
            config=cfg,
            seed=trainer.args.seed,
            subset_manifest=str(manifest) if manifest.exists() else None,
            subset_manifest_sha256=sk.sha256(manifest) if manifest.exists() else None,
            init_weights=str(weights),
            init_weights_sha256=sk.sha256(weights) if weights.is_file() else None,
        )
        if config_path:
            shutil.copy2(config_path, save_dir / "train_config.yaml")
        (save_dir / "pip_freeze.txt").write_text(pip_freeze())
        
        # One entry per session -> You can see exactly when a Colab disconnect + resume happend
        meta["sessions"].append({
            "start_epoch": trainer.start_epoch,
            "resumed": bool(trainer.args.resume),
            "effective_args": {k: str(v) for k, v in vars(trainer.args).items()},
            **environment_info(),
        })
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        
def check_checkpoint(weights: Path, data_yaml: str | Path, imgsz: int, max_det: int) -> int:
    """Phase 4 checkpoint test: best.pt reloads and predicts boxes on a validation image."""
    img = val_image_paths(data_yaml)[0]
    res = YOLO(str(weights)).predict(str(img), imgsz=imgsz, max_det=max_det, verbose=False)[0]
    assert res.boxes.xyxy.ndim == 2 and res.boxes.xyxy.shape[1] == 4
    return len(res.boxes)

def run_training(cfg: dict, *, config_path=None, smoke=False, overrides=None, resume=False, extra_callbacks: dict | None = None) -> Path:
    """Returns the run directory. Seperate from main() so tests can call it."""
    cfg = dict(cfg, data=str(resolve_data_yaml(cfg["data"])))
    train_args = build_train_args(cfg, smoke, overrides)
    run_dir = Path(train_args["project"]) / train_args["name"]
    last = run_dir / "weights" / "last.pt"
    
    if resume: 
        if not last.exists():
            raise SystemExit(f"nothing to resume: {last} not found")
        model = YOLO(str(last))
    else:
        if run_dir.exists():
            # stops Ultralytics from quietly creating name2, name3...
            raise SystemExit(f"{run_dir} already exists: use -- resume or a new --name")
        model = YOLO(cfg["model"])
    
    model.add_callback("on_train_start", lambda tr: write_metadata(tr, cfg, config_path))
    for event, fn in (extra_callbacks or {}).items():
        model.add_callback(event, fn)
        
    if resume:
        # Ultralytics restores ALL training args (epochs, lr schedule, data, save_dir, ...) plus
        # the optimizer/EMA state from last.pt. Only a few hardware-type overrides
        # (e.g. batch, device, workers; this varies by version) are honoured
        model.train(resume=True, **(overrides or {}))
    else:
        model.train(**train_args)
        
    best = run_dir / "weights" / "best.pt"
    n = check_checkpoint(best, cfg["data"], train_args["imgsz"], train_args["max_det"])
    print(f"best.pt reloads OK; {n} boxes on first val image. Run dir: {run_dir}")
    return run_dir

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data")
    ap.add_argument("--project")
    ap.add_argument("--name")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    
    cfg = load_config(a.config)
    for key in ("data", "project", "name"):
        if getattr(a, key):
            cfg[key] = getattr(a, key)
    run_training(cfg, config_path=a.config, smoke=a.smoke,
                 overrides=parse_overrides(a.set), resume=a.resume)

if __name__ == "__main__":      # required on Windows: dataloader workers re-import this module
    main()