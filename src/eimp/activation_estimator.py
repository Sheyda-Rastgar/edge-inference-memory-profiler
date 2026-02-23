import numpy as np

# ONNX tensor type enum → bytes per element
DTYPE_SIZE = {
    1: 4,   # float32
    10: 2,  # float16
    3: 1,   # int8
}


def tensor_size_bytes(tensor_type):
    """Estimate tensor size in bytes from ONNX tensor_type."""
    if not tensor_type.HasField("shape"):
        return 0

    dims = []
    for dim in tensor_type.shape.dim:
        if dim.dim_value > 0:
            dims.append(dim.dim_value)
        else:
            return 0  # skip dynamic shapes in v0.1

    if not dims:
        return 0

    element_count = int(np.prod(dims))
    dtype = tensor_type.elem_type
    bytes_per_elem = DTYPE_SIZE.get(dtype, 4)

    return element_count * bytes_per_elem