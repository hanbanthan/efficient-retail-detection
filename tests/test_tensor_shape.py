import torch
from src.fundamentals.tensor_shapes import build_tiny_cnn, conv_out_size


def test_formula_matches_pytorch():
    for size in (224, 225, 640):
        conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
        out = conv(torch.randn(1, 3, size, size))
        assert out.shape[-1] == conv_out_size(size, k=3, s=2, p=1)


def test_tiny_cnn_output_shape():
    y = build_tiny_cnn()(torch.randn(2, 3, 224, 224))
    assert tuple(y.shape) == (2, 64, 28, 28)