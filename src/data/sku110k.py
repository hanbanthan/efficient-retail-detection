"""SKU-110K annotation utilities: parse, validate, sample, summarize, convert to YOLO.

Official CSV format (no header), one row per box:
    image_name,x1,y1,x2,y2,class,image_width,image_height
    
Coordinates are absolute pixels (xyxy). The only class is "object".

"""

import csv
import hashlib
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
from src.fundamentals.boxes import box_area, cxcywh_to_xyxy, xyxy_to_cxcywh

SPLITS = ("train", "val", "test")
CLASS_NAME = "object"
CLASS_ID = 0

@dataclass
class ImageAnn:
    name: str
    width: int
    height: int
    boxes: np.ndarray # (N, 4) float, xyxy pixels
    classes: list[str] = field(default_factory=list) 
    n_size_variants: int = 1 # >1 means rows disagree on the image size
    
    def __len__(self) -> int:
        return len(self.boxes)
    
def csv_path(root: str | Path, split: str) -> Path:
    return Path(root) / "annotations" / f"annotations_{split}.csv"

def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# parsing 
def read_annotations(path: str | Path) -> dict[str, ImageAnn]:
    """Parse one official CSV into {image_name: ImageAnn}."""
    groups = defaultdict(lambda: {"boxes": [], "classes": [], "sizes": set()})
    with open(path, newline="", encoding="utf-8") as f:
        for line_no, row in enumerate(csv.reader(f), start=1):
            if not row:
                continue
            
            if line_no == 1 and row == [
                "image_name", "x1", "y1", "x2", "y2",
                "class", "image_width", "image_height",
            ]:
                continue
            
            if len(row) != 8:
                raise ValueError(f"{path}:{line_no}: expected 8 columns, got {len(row)}")

            g = groups[row[0].strip()]
            g["boxes"].append([float(v) for v in row[1:5]])
            g["classes"].append(row[5].strip())
            g["sizes"].add((int(row[6]), int(row[7])))
    out = {}
    for name, g in groups.items():
        w, h = sorted(g["sizes"])[0]
        out[name] = ImageAnn(name, w, h, np.asarray(g["boxes"], float).reshape(-1, 4),
                             g["classes"], len(g["sizes"]))
    return out
    
# validation
def validate_image(ann: ImageAnn, tol: float = 0.0) -> list[str]:
    """Return problems for one image (empty list == valid)."""
    b = ann.boxes
    if len(b) == 0:
        return ["no boxes"]
    problems = []
    if ann.n_size_variants != 1:
        problems.append(f"{ann.n_size_variants} different image sizes across rows")
    if ann.width <= 0 or ann.height <= 0:
        problems.append(f"invalid image size {ann.width}x{ann.height}")
    finite = np.isfinite(b).all(axis=1)
    if not finite.all():
        problems.append(f"{int((~finite).sum())} boxes with NaN/inf")
    degenerate = (b[:, 2] <= b[:, 0]) | (b[:, 3] <= b[:, 1])
    if degenerate.any():
        problems.append(f"{int(degenerate.sum())} boxes with zero/negative width or height")
    outside = ((b[:, 0] < -tol) | (b[:, 1] < -tol)
               | (b[:, 2] > ann.width + tol) | (b[:, 3] > ann.height + tol))
    if outside.any():
        problems.append(f"{int(outside.sum())} boxes outside the image (tol={tol}px)")
    bad = sorted(set(ann.classes) - {CLASS_NAME})
    if bad:
        problems.append(f"unsupported classes {bad}")
    return problems

def validate_split(anns: dict[str, ImageAnn], tol: float = 0.0) -> dict[str, list[str]]:
    return {
        name: p for name, a in anns.items() if (p := validate_image(a, tol))
    }

def check_image_file(path: str | Path, expected_wh: tuple[int, int] | None = None) -> str | None:
    """None if the file fully decodes (and matches the annotated size), else a reason.
    Note: PIL reports raw pixel size; a swapped WxH usually means EXIF rotation."""
    
    path = Path(path)
    if not path.is_file():
        return "missing"
    try:
        with Image.open(path) as im:
            im.load()                    # full decode catches truncated/corrupt JPEGs
            size = im.size
    except Exception as e:  # noqa: BLE001
        return f"corrupt: {e}"
    if expected_wh is not None and tuple(size) != tuple(expected_wh):
        return f"size {size} != annotated {tuple(expected_wh)}"
    return None

# deterministic sampling 
def deterministic_order(names, seed: int) -> list[str]:
    """Seeded permutation of the *sorted* names: independent of CSV row order.
    (Store the chosen IDs anyway; RNG streams are not guaranteed across numpy versions.)"""
    
    pool = sorted(names)
    perm = np.random.default_rng(seed).permutation(len(pool))
    return [pool[i] for i in perm]

# statistics
def _describe(x) -> dict:
    x = np.asarray(x, float)
    if x.size == 0:
        return {}
    keys = ["min", "p5", "p25", "median", "p75", "p95", "max"]
    return {"mean": float(x.mean()),
            **dict(zip(keys, map(float, np.percentile(x, [0, 5, 25, 50, 75, 95, 100]))))}
    
def summarize(anns: dict[str, ImageAnn], train_size: int = 640) -> dict:
    items = list(anns.values())
    boxes = np.concatenate([a.boxes for a in items])
    area = box_area(boxes)
    img_area = np.concatenate([np.full(len(a), max(a.width * a.height, 1), float) for a in items])
    scale = np.concatenate([np.full(len(a), train_size / max(a.width, a.height, 1)) for a in items])
    side_at = np.sqrt(area) * scale      # box side length after resizing to the training size
    sizes = Counter((a.width, a.height) for a in items)
    return {
        "images": len(items),
        "boxes": int(len(boxes)),
        "objects_per_image": _describe([len(a) for a in items]),
        "box_area_px": _describe(area),
        "box_area_frac_of_image": _describe(area / img_area),
        f"box_side_px_at_{train_size}": _describe(side_at),
        f"frac_boxes_side_lt_8px_at_{train_size}": float((side_at < 8).mean()),
        f"frac_boxes_side_lt_32px_at_{train_size}": float((side_at < 32).mean()),
        "unique_image_sizes": len(sizes),
        "top_image_sizes": [[w, h, c] for (w, h), c in sizes.most_common(10)],
        "classes": dict(Counter(c for a in items for c in a.classes)),
    }
    
# YOLO conversion
def clipped_boxes(ann: ImageAnn) -> np.ndarray:
    """Boxes clipped to the image, with non-finite/degenerate ones removed (pixels, xyxy)."""
    b = ann.boxes.copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, ann.width)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, ann.height)
    keep = np.isfinite(b).all(axis=1) & (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    return b[keep]

def to_yolo(ann: ImageAnn) -> np.ndarray:
    """(N, 5) rows [cls, cx, cy, w, h], normalized to [0, 1]."""
    c = xyxy_to_cxcywh(clipped_boxes(ann)).reshape(-1, 4)
    c = c / np.array([ann.width, ann.height] * 2, float)
    return np.hstack([np.full((len(c), 1), CLASS_ID, float), c])

def from_yolo(labels, width: int, height: int) -> np.ndarray:
    labels = np.asarray(labels, float).reshape(-1, 5)
    return cxcywh_to_xyxy(labels[:, 1:] * np.array([width, height] * 2, float)).reshape(-1, 4)

def write_label_file(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n"
                            for r in labels))
    
def read_label_file(path: Path) -> np.ndarray:
    if not Path(path).read_text().strip():
        return np.zeros((0, 5))
    return np.loadtxt(path, ndmin=2).reshape(-1, 5)

def label_path(out_root: Path, split: str, image_name: str) -> Path:
    return Path(out_root) / "labels" / split / f"{Path(image_name).stem}.txt"

def export_yolo_split(anns, ids, image_dir, out_root, split) -> dict:
    """Copy images + write YOLO labels in Ultralytics layout (images/<split>, labels/<split>)."""
    img_out = Path(out_root) / "images" / split
    img_out.mkdir(parents=True, exist_ok=True)
    dropped = 0
    for name in ids:
        dst = img_out / name
        if not dst.exists():
            shutil.copy2(Path(image_dir) / name, dst)
        labels = to_yolo(anns[name])
        dropped += len(anns[name]) - len(labels)
        write_label_file(label_path(out_root, split, name), labels)
    return {"images": len(ids), "dropped_boxes": dropped}

def max_roundtrip_error(anns, ids, out_root, split) -> float:
    """Max |pixel difference| between CSV boxes (clipped) and boxes decoded from label files."""
    worst = 0.0
    for name in ids:
        a = anns[name]
        got = from_yolo(read_label_file(label_path(out_root, split, name)), a.width, a.height)
        exp = clipped_boxes(a)
        if got.shape != exp.shape:
            return float("inf")
        worst = max(worst, float(np.abs(got - exp).max(initial=0.0)))
    return worst

def write_data_yaml(out_root: str | Path) -> Path:
    out_root = Path(out_root)
    p = out_root / "data.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(out_root.resolve()), "train": "images/train",
                        "val": "images/val", "names": {CLASS_ID: CLASS_NAME}}, f, sort_keys=False)
    return p