import argparse

from .onnx_loader import load_model
from .shape_inference import infer_shapes
from .peak_analyzer import compute_memory_report
from .report import print_report


def main():
    parser = argparse.ArgumentParser(
        description="Edge Inference Memory Profiler"
    )
    parser.add_argument(
        "model",
        help="Path to ONNX model file",
    )

    args = parser.parse_args()

    # Load ONNX model
    model = load_model(args.model)

    # Run shape inference (required for tensor size estimation)
    model = infer_shapes(model)

    # Compute memory metrics
    (
        weight_bytes,
        naive_peak_bytes,
        reuse_peak_bytes,
        _activations,
        _allocations,
        meta,
    ) = compute_memory_report(model)

    print_report(weight_bytes, naive_peak_bytes, reuse_peak_bytes, meta)