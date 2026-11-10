"""Fast tests always run. Slow tests (real Ultralytics training on CPU) need RUN_SLOW=1.
Requires the `fake_root` fixture to live in tests/conftest.py."""
import os

import numpy as np
import pytest
import yaml

from src.training.common import parse_overrides, resolve_data_yaml
from src.training.inspect_run import diagnose, read_results
from src.training.train import build_train_args

CFG = {"model": "yolo11n.pt", "data": "d/data.yaml", "project": "runs", "name": "exp",
       "train": {"epochs": 50, "batch": 16, "imgsz": 640, "max_det": 1000},
       "smoke": {"epochs": 1, "fraction": 0.05}}


# ── config handling ────────────────────────────────────────────────────────
def test_parse_overrides_types():
    assert parse_overrides(["epochs=3", "cache=false", "lr0=0.01", "device=cpu"]) == \
        {"epochs": 3, "cache": False, "lr0": 0.01, "device": "cpu"}
    with pytest.raises(ValueError):
        parse_overrides(["epochs"])


def test_build_train_args_priority():
    a = build_train_args(CFG, smoke=True, overrides={"batch": 2})
    assert a["epochs"] == 1 and a["fraction"] == 0.05 and a["batch"] == 2
    assert a["name"] == "exp_smoke" and a["exist_ok"] is False and a["max_det"] == 1000
    assert build_train_args(CFG)["epochs"] == 50
    assert "fraction" not in build_train_args(CFG)


def test_resolve_data_yaml_makes_path_portable(tmp_path):
    p = tmp_path / "data.yaml"
    p.write_text(yaml.safe_dump({"path": "D:/somewhere/else", "train": "images/train",
                                 "val": "images/val", "names": {0: "object"}}))
    cfg = yaml.safe_load(resolve_data_yaml(p).read_text())
    assert cfg["path"] == str(tmp_path.resolve()) and cfg["names"] == {0: "object"}


# ── curve diagnosis ────────────────────────────────────────────────────────
def curves(m, val, tr):
    n = len(m)
    r = {"epoch": np.arange(1, n + 1, dtype=float), "metrics/mAP50-95(B)": np.array(m, float)}
    for l in ("box_loss", "cls_loss", "dfl_loss"):
        r[f"val/{l}"] = np.array(val, float) / 3
        r[f"train/{l}"] = np.array(tr, float) / 3
    return r


def test_diagnose_underfit_and_overfit():
    n = 12
    rising = np.linspace(0.1, 0.5, n)
    falling = np.linspace(3, 1, n)
    assert any("UNDER" in s for s in diagnose(curves(rising, falling, falling)))
    val_u = np.r_[np.linspace(3, 1, 6), np.linspace(1.1, 1.6, 6)]      # val loss turns up
    peaked = np.r_[np.linspace(0.1, 0.4, 6), np.linspace(0.39, 0.3, 6)]
    assert any("OVERFITTING" in s for s in diagnose(curves(peaked, val_u, falling)))


def test_read_results_strips_padded_headers(tmp_path):
    p = tmp_path / "results.csv"
    p.write_text("   epoch,  train/box_loss\n1, 2.5\n2, 2.0\n")
    r = read_results(p)
    assert r["epoch"].tolist() == [1, 2] and r["train/box_loss"].tolist() == [2.5, 2.0]


# ── slow: real training on the fake dataset (CPU, tiny images, no downloads) ──
slow = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1")


def tiny_cfg(fake_root, tmp_path):
    pytest.importorskip("ultralytics")
    from src.data.make_subset import build_subset
    out = tmp_path / "subset"
    build_subset(fake_root, out, n_train=4, n_val=2, seed=0)
    # yolo11n.yaml = same architecture with random weights -> nothing to download
    return {"model": "yolo11n.yaml", "data": str(out / "data.yaml"),
            "project": str(tmp_path / "runs"), "name": "t",
            "train": {"epochs": 2, "imgsz": 64, "batch": 2, "device": "cpu", "workers": 0,
                      "seed": 0, "max_det": 1000, "plots": False, "amp": False}}


@slow
def test_train_then_reload_best(fake_root, tmp_path):
    from src.training.train import run_training
    run = run_training(tiny_cfg(fake_root, tmp_path))
    assert (run / "weights" / "best.pt").exists() and (run / "results.csv").exists()
    meta = yaml.safe_load((run / "run_meta.json").read_text())     # JSON is valid YAML
    assert meta["seed"] == 0 and len(meta["sessions"]) == 1


@slow
def test_interrupt_then_resume(fake_root, tmp_path):
    """Simulate a Colab disconnect: crash after epoch 1 (last.pt already saved), then resume."""
    from src.training.train import run_training

    class Disconnect(Exception):
        pass

    def crash(trainer):          # on_fit_epoch_end fires AFTER last.pt is written
        if trainer.epoch == 0:
            raise Disconnect

    cfg = tiny_cfg(fake_root, tmp_path)
    with pytest.raises(Disconnect):
        run_training(cfg, extra_callbacks={"on_fit_epoch_end": crash})
    run = run_training(cfg, resume=True)
    rows = read_results(run / "results.csv")
    assert rows["epoch"].tolist()[-1] == 2                         # finished both epochs
    meta = yaml.safe_load((run / "run_meta.json").read_text())
    assert [s["resumed"] for s in meta["sessions"]] == [False, True]