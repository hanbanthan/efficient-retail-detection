"""Create a deterministic small train/val subset in Ultralytics YOLO layout.
    python -m src.data.make_subset --root D:\efficient-retail-detection\SKU110K_fixed --out D:\efficient-retail-detection\sku110k_subset --n-train 400 --n-val 100 --seed 0
Selection rule (also written to manifest.json):
  1. pool = images of the OFFICIAL split whose annotations pass validate_image(tol)
  2. order = seeded permutation of the sorted pool
  3. take the first n images in that order whose file decodes and matches the annotated size
The test split is never read.  Commit manifest.json (IDs only); never commit images.
"""

from __future__ import annotations
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from src.data import sku110k as sk

def build_subset(root, out, n_train: int, n_val: int, seed: int = 0, tol: float = 0.0) -> dict:
    root, out = Path(root), Path(out)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": str(root), "seed": seed, "tol_px": tol,
        "rule": "valid-annotation pool -> default_rng(seed).permutation(sorted pool) -> first n decodable",
        "python": platform.python_version(), "numpy": np.__version__, "splits": {},
    }
    for split, n in (("train", n_train), ("val", n_val)):
        path = sk.csv_path(root, split)
        anns = sk.read_annotations(path)
        invalid = sk.validate_split(anns, tol)
        pool = [k for k in anns if k not in invalid]
        chosen, skipped = [], {}
        for name in sk.deterministic_order(pool, seed):
            a = anns[name]
            reason = sk.check_image_file(root / "images" / name, (a.width, a.height))
            if reason:
                skipped[name] = reason
            elif len(chosen) < n:
                chosen.append(name)
        if len(chosen) < n:
            raise RuntimeError(f"{split}: only {len(chosen)} usable images, requested {n}")
        chosen.sort()
        info = sk.export_yolo_split(anns, chosen, root / "images", out, split)
        err = sk.max_roundtrip_error(anns, chosen, out, split)
        if err > 0.5:
            raise RuntimeError(f"{split}: YOLO round-trip error {err:.3f}px > 0.5px")
        sub = {k: anns[k] for k in chosen}
        manifest["splits"][split] = {
            "csv_sha256": sk.sha256(path),
            "full_split_images": len(anns),
            "excluded_invalid_annotations": sorted(invalid),
            "skipped_files": skipped,
            "n_images": len(chosen),
            "n_boxes": sum(len(a) for a in sub.values()),
            "dropped_boxes_on_export": info["dropped_boxes"],
            "max_roundtrip_error_px": err,
            "objects_per_image": sk.summarize(sub)["objects_per_image"],
            "image_ids": chosen,
        }
        print(f"[{split}] {len(chosen)} images, {manifest['splits'][split]['n_boxes']} boxes, "
              f"round-trip err {err:.4f}px, skipped {len(skipped)}, invalid {len(invalid)}")
    sk.write_data_yaml(out)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"data.yaml + manifest.json -> {out}")
    return manifest

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.0)
    a = ap.parse_args()
    build_subset(a.root, a.out, a.n_train, a.n_val, a.seed, a.tol)
    
if __name__ == "__main__":
    main()