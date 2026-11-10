"""Inspect, validate and summarize all official SKU-110K splits -> one JSON report.
    python -m src.data.inspect_dataset --root D:\efficient-retail-detection\SKU110K_fixed --out experiments/data_report.json
    python -m src.data.inspect_dataset --root D:\efficient-retail-detection\SKU110K_fixed --check-images   # decode every JPEG (slow)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.data import sku110k as sk

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", default="experiments/data_report.json", type=Path)
    ap.add_argument("--tol", default=0.0, type=float, help="out-of-bounds tolerance (px)")
    ap.add_argument("--check-images", action="store_true")
    args = ap.parse_args()
    print("First raw rows of the train CSV:")
    with open(sk.csv_path(args.root, "train"), encoding="utf-8") as f:
        for _ in range(3):
            print("  ", f.readline().rstrip())
    report: dict = {"root": str(args.root), "tol_px": args.tol, "splits": {}}
    names = {}
    for split in sk.SPLITS:
        path = sk.csv_path(args.root, split)
        anns = sk.read_annotations(path)
        names[split] = set(anns)
        problems = sk.validate_split(anns, args.tol)
        entry = {
            "csv": str(path),
            "csv_sha256": sk.sha256(path),
            "names_have_split_prefix": all(n.startswith(f"{split}_") for n in anns),
            **sk.summarize(anns),
            "images_with_box_problems": len(problems),
            "box_problem_examples": dict(list(problems.items())[:20]),
        }
        if args.check_images:
            bad = {}
            for a in anns.values():
                reason = sk.check_image_file(args.root / "images" / a.name, (a.width, a.height))
                if reason:
                    bad[a.name] = reason
            entry["image_file_problems"] = bad
        report["splits"][split] = entry
        opi = entry["objects_per_image"]
        print(f"\n[{split}] {entry['images']} images, {entry['boxes']} boxes, classes={entry['classes']}")
        print(f"  objects/image: min={opi['min']:.0f} median={opi['median']:.0f} "
              f"mean={opi['mean']:.1f} max={opi['max']:.0f}")
        print(f"  image sizes: {entry['unique_image_sizes']} unique, top={entry['top_image_sizes'][:3]}")
        print(f"  boxes with side < 32px at 640: {entry['frac_boxes_side_lt_32px_at_640']:.1%}")
        print(f"  images with box problems: {len(problems)}")
        if args.check_images:
            print(f"  missing/corrupt/size-mismatch files: {len(entry['image_file_problems'])}")
    report["split_overlap"] = {f"{a}&{b}": len(names[a] & names[b])
                               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    on_disk = {p.name for p in (args.root / "images").glob("*.jpg")}
    annotated = set().union(*names.values())
    report["images_on_disk"] = len(on_disk)
    report["annotated_but_not_on_disk"] = sorted(annotated - on_disk)
    report["on_disk_but_not_annotated"] = sorted(on_disk - annotated)
    print(f"\nsplit overlap: {report['split_overlap']}")
    print(f"on disk: {len(on_disk)}, annotated-but-missing: {len(report['annotated_but_not_on_disk'])}, "
          f"unannotated: {len(report['on_disk_but_not_annotated'])}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"report -> {args.out}")
if __name__ == "__main__":
    main()