"""Print tensor shapes through a tiny CNN and verify them against the formula."""
import torch
import torch.nn as nn

def conv_out_size(size: int, k: int, s: int = 1, p: int = 0, d: int = 1) -> int:
    return (size + 2 * p - d * (k -1) -1) // s + 1

def build_tiny_cnn() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU()
    )

def trace_shapes(model: nn.Sequential, x: torch.Tensor) -> list[tuple[str, tuple]]:
    rows = [("input", tuple(x.shape))]
    for i, layer in enumerate(model):
        x = layer(x)
        rows.append((f"{i}: {layer.__class__.__name__}", tuple(x.shape)))
    return rows

if __name__ == "__main__":
    model = build_tiny_cnn().eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for name, shape in trace_shapes(model, x):
            print(f"{name:<18} {shape}")
    print("total stride:", 224 // trace_shapes(model, x)[-1][1][-1])
    