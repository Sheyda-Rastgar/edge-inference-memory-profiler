import onnx


def load_model(path: str):
    """Load ONNX model from disk."""
    return onnx.load(path)