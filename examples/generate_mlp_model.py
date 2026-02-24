import os
from pathlib import Path

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

    def forward(self, x):
        return self.net(x)


def main():
    project_root = Path(__file__).resolve().parent.parent
    examples_dir = project_root / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    model = MLP()
    model.eval()

    dummy = torch.randn(1, 256)
    export_path = examples_dir / "mlp.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(export_path),
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
    )

    print(f"Exported to {export_path}")


if __name__ == "__main__":
    main()