import onnx


def infer_shapes(model):
    """Run ONNX shape inference."""
    return onnx.shape_inference.infer_shapes(model)