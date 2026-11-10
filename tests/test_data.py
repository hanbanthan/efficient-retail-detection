import csv
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from src.data import sku110k as sk
from src.data.make_subset import build_subset

W, H = 100, 80


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


@pytest.fixture
def fake_root(tmp_path) -> Path:
    """Tiny SKU-110K-shaped dataset; image <split>_<i>.jpg has i+1 boxes."""
    root = tmp_path / "sku"
    (root / "images").mkdir(parents=True)
    for split, n in (("train", 6), ("val", 3), ("test", 2)):
        rows = []
        for i in range(n):
            name = f"{split}_{i}.jpg"
            Image.new("RGB", (W, H), (20 * i, 100, 150)).save(root / "images" / name)
            rows += [[name, 10 + 5 * k, 10, 20 + 5 * k, 40, "object", W, H] for k in range(i + 1)]
        write_csv(sk.csv_path(root, split), rows)
    return root


def ann(boxes, classes=None, w=W, h=H) -> sk.ImageAnn:
    b = np.asarray(boxes, float).reshape(-1, 4)
    return sk.ImageAnn("x.jpg", w, h, b, classes or ["object"] * len(b))


# ── parsing ────────────────────────────────────────────────────────────────
def test_read_groups_rows_by_image(fake_root):
    anns = sk.read_annotations(sk.csv_path(fake_root, "train"))
    assert len(anns) == 6
    assert len(anns["train_2.jpg"]) == 3
    assert anns["train_0.jpg"].boxes.tolist() == [[10, 10, 20, 40]]
    assert (anns["train_0.jpg"].width, anns["train_0.jpg"].height) == (W, H)


def test_header_tolerated_and_bad_rows_rejected(tmp_path):
    p = tmp_path / "a.csv"
    write_csv(p, [["image_name", "x1", "y1", "x2", "y2", "class", "image_width", "image_height"],
                  ["a.jpg", 0, 0, 5, 5, "object", 10, 10]])
    assert len(sk.read_annotations(p)) == 1
    write_csv(p, [["a.jpg", 0, 0, 5]])
    with pytest.raises(ValueError, match="expected 8 columns"):
        sk.read_annotations(p)


# ── validation ─────────────────────────────────────────────────────────────
def test_clean_split_has_no_problems(fake_root):
    for split in sk.SPLITS:
        assert sk.validate_split(sk.read_annotations(sk.csv_path(fake_root, split))) == {}


@pytest.mark.parametrize("box,cls,expected", [
    ([10, 10, 10, 20], "object", "zero/negative"),      # zero width
    ([30, 10, 20, 20], "object", "zero/negative"),      # inverted x
    ([-5, 0, 10, 10], "object", "outside"),
    ([0, 0, W + 1, 10], "object", "outside"),
    ([np.nan, 0, 10, 10], "object", "NaN"),
    ([0, 0, 10, 10], "bottle", "unsupported classes"),
])
def test_validator_catches_problems(box, cls, expected):
    problems = sk.validate_image(ann([box], [cls]))
    assert any(expected in p for p in problems), problems


def test_validator_tolerance_and_empty():
    assert sk.validate_image(ann([[-1, 0, 10, 10]]), tol=2) == []
    assert sk.validate_image(ann(np.zeros((0, 4)))) == ["no boxes"]


# ── sampling ───────────────────────────────────────────────────────────────
def test_order_is_deterministic_and_input_order_independent():
    names = [f"train_{i}.jpg" for i in range(50)]
    a = sk.deterministic_order(names, seed=0)
    assert a == sk.deterministic_order(list(reversed(names)), seed=0)
    assert a != sk.deterministic_order(names, seed=1)
    assert sorted(a) == sorted(names)


# ── YOLO conversion ────────────────────────────────────────────────────────
def test_yolo_roundtrip_random_boxes():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 60, (100, 2))
    boxes = np.hstack([xy, xy + rng.uniform(1, 20, (100, 2))])
    a = ann(boxes)
    lab = sk.to_yolo(a)
    assert (lab[:, 0] == 0).all() and (lab[:, 1:] >= 0).all() and (lab[:, 1:] <= 1).all()
    assert np.allclose(sk.from_yolo(lab, W, H), boxes)


def test_yolo_clips_and_drops_degenerate():
    lab = sk.to_yolo(ann([[-10, -10, 20, 20], [W + 5, 0, W + 10, 10]]))
    assert len(lab) == 1                                  # fully-outside box dropped
    assert np.allclose(sk.from_yolo(lab, W, H), [[0, 0, 20, 20]])


def test_label_file_roundtrip(tmp_path):
    lab = sk.to_yolo(ann([[10, 10, 30, 40], [50, 20, 70, 60]]))
    p = tmp_path / "l.txt"
    sk.write_label_file(p, lab)
    assert np.allclose(sk.read_label_file(p), lab, atol=1e-6)
    sk.write_label_file(p, np.zeros((0, 5)))
    assert sk.read_label_file(p).shape == (0, 5)


# ── image files ────────────────────────────────────────────────────────────
def test_check_image_file(tmp_path):
    good = tmp_path / "good.jpg"
    Image.new("RGB", (W, H)).save(good)
    assert sk.check_image_file(good, (W, H)) is None
    assert sk.check_image_file(tmp_path / "nope.jpg") == "missing"
    assert "size" in sk.check_image_file(good, (H, W))
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(good.read_bytes()[:200])           # truncated JPEG
    assert sk.check_image_file(broken).startswith("corrupt")


# ── statistics ─────────────────────────────────────────────────────────────
def test_summarize(fake_root):
    s = sk.summarize(sk.read_annotations(sk.csv_path(fake_root, "train")))
    assert s["images"] == 6 and s["boxes"] == 21          # 1+2+...+6
    assert s["objects_per_image"]["min"] == 1 and s["objects_per_image"]["max"] == 6
    assert s["box_area_px"]["median"] == 300              # 10 x 30
    assert s["classes"] == {"object": 21}
    assert s["top_image_sizes"][0][:2] == [W, H]


# ── end-to-end subset ──────────────────────────────────────────────────────
def test_build_subset_end_to_end(fake_root, tmp_path):
    out = tmp_path / "subset"
    m = build_subset(fake_root, out, n_train=4, n_val=2, seed=0)
    assert m == {**m}  # manifest is a plain dict
    for split, n in (("train", 4), ("val", 2)):
        ids = m["splits"][split]["image_ids"]
        assert len(ids) == n and all(i.startswith(f"{split}_") for i in ids)
        assert len(list((out / "images" / split).glob("*.jpg"))) == n
        assert len(list((out / "labels" / split).glob("*.txt"))) == n
        assert m["splits"][split]["max_roundtrip_error_px"] < 1e-3
    assert not set(m["splits"]["train"]["image_ids"]) & set(m["splits"]["val"]["image_ids"])
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert cfg["names"] == {0: "object"}
    m2 = build_subset(fake_root, tmp_path / "subset2", n_train=4, n_val=2, seed=0)
    assert m2["splits"]["train"]["image_ids"] == m["splits"]["train"]["image_ids"]


def test_build_subset_skips_corrupt_file(fake_root, tmp_path):
    (fake_root / "images" / "val_1.jpg").write_bytes(b"not a jpeg")
    m = build_subset(fake_root, tmp_path / "s", n_train=1, n_val=2, seed=0)
    assert "val_1.jpg" not in m["splits"]["val"]["image_ids"]
    assert "val_1.jpg" in m["splits"]["val"]["skipped_files"]


def test_ultralytics_loads_subset(fake_root, tmp_path):
    pytest.importorskip("ultralytics")
    from src.data.check_ultralytics import load_dataset
    out = tmp_path / "subset"
    build_subset(fake_root, out, n_train=4, n_val=3, seed=0)
    ds = load_dataset(str(out / "data.yaml"), "val", imgsz=128)
    assert len(ds) == 3
    assert sorted(len(l["cls"]) for l in ds.labels) == [1, 2, 3]
    item = ds[0]
    assert item["img"].shape[0] == 3 and item["bboxes"].shape[1] == 4


# ── optional: real dataset smoke test (set SKU110K_ROOT) ───────────────────
REAL = os.environ.get("SKU110K_ROOT")


@pytest.mark.skipif(not REAL, reason="set SKU110K_ROOT to the extracted SKU110K_fixed folder")
def test_real_val_split_smoke():
    root = Path(REAL)
    anns = sk.read_annotations(sk.csv_path(root, "val"))
    assert len(anns) == 588                               # documented val size
    assert {c for a in anns.values() for c in a.classes} == {"object"}
    assert len(sk.validate_split(anns, tol=1.0)) / len(anns) < 0.05
    for name in sorted(anns)[:20]:
        a = anns[name]
        assert sk.check_image_file(root / "images" / name, (a.width, a.height)) is None, name