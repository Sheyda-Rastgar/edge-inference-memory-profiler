import os
from pathlib import Path

import torch
import torch.nn as nn


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        return self.fc(x)


def main():
    # Get project root (parent of examples folder)
    project_root = Path(__file__).resolve().parent.parent
    examples_dir = project_root / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    model = TinyModel()
    model.eval()  # Set to inference mode

    dummy_input = torch.randn(1, 16)

    export_path = examples_dir / "tiny.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(export_path),
        opset_version=17,
        do_constant_folding=True,
        export_params=True
    )

    print(f"Model exported to {export_path}")


if __name__ == "__main__":
    main()