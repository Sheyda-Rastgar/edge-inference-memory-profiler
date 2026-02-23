import argparse

from .onnx_loader import load_model
from .shape_inference import infer_shapes
from .peak_analyzer import compute_naive_peak
from .report import print_report


def main():
    parser = argparse.ArgumentParser(
        description="Edge Inference Memory Profiler"
    )
    parser.add_argument("model", help="Path to ONNX model file")

    args = parser.parse_args()

    model = load_model(args.model)
    model = infer_shapes(model)

    peak, activations, weight_bytes = compute_naive_peak(model)
    print_report(peak, activations, weight_bytes)