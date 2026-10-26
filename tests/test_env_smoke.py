import platform
import numpy as np
import onnxruntime as ort
import torch
import torchvision


def test_python_and_core_imports():
    """Verify Python version and core numerical libraries load cleanly."""
    major, minor, _ = platform.python_version_tuple()
    assert int(major) == 3 and int(minor) >= 10, f"Expected Python >=3.10, got {platform.python_version()}"

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tensor = torch.from_numpy(arr)
    assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))


def test_onnxruntime_cpu_provider_available():
    """Verify ONNX Runtime has CPUExecutionProvider available for local benchmarking."""
    providers = ort.get_available_providers()
    assert "CPUExecutionProvider" in providers, f"CPUExecutionProvider missing from {providers}"


def test_torchvision_load():
    """Verify TorchVision imports without C++/CUDA symbol mismatches."""
    assert hasattr(torchvision.ops, "nms")