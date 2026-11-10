"""Shared helpers: config loading, CLI overrides, portable data.yaml, environment capture."""

from __future__ import annotations
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import yaml

REQUIRED_KEYS = ("model", "data", "project", "name", "train")

def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"{path}: missing keys {missing}")
    return cfg

def parse_overrides(items: list[str] | None) -> dict:
    """['epochs=3', 'cache=false', 'batch=8'] -> {'epochs': 3, 'cache': False, 'batch': 8}.
    Values are parsed as YAML so numbers and booleans get the right type."""
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"override must look like key=value, got{item!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out

def resolve_data_yaml(data_yaml: str | Path) -> Path:
    """Write data.resolved.yaml next to data.yaml, with `path` set to that folder.
    
    make_subset writes an ABSOLUTE `path` (e.g. D:/efficient-retail-detection/...). After you zip the subset
    and unzip it on Colab (/content/...), that path is wrong. This makes the folder portable.
    
    Keep the folder at the SAME location in every Colab session: resume re-reads this path.
    """
    data_yaml = Path(data_yaml).resolve()
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    cfg["path"] = str(data_yaml.parent)
    out = data_yaml.with_name("data.resolved.yaml")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out

def val_image_paths(data_yaml: str | Path) -> list[Path]:
    """Sorted (=deterministic) list of validation images from a (resolved) data.yaml."""
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    val_dir = Path(cfg["path"]) / cfg["val"]
    
    return sorted(val_dir.glob("*.jpg"))

def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None

def git_commit() -> str | None:
    """Current commit, with '-dirty' if there are uncommitted changes (None outside a repo)."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None 

def environment_info() -> dict:
    """Everything needed to say *where* and *with what* a result was produced."""
    import torch
    cuda = torch.cuda.is_available()
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: _version(p) for p in
                     ("torch", "torchvision", "ultralytics", "numpy", "pillow",
                      "opencv-python", "opencv-python-headless")},
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "cpu": platform.processor() or None,
        "git_commit": git_commit(),
        "argv": sys.argv, 
    }
    
def pip_freeze() -> str:
    return subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    