import numpy as np

# ONNX TensorProto DataType enum -> bytes per element
DTYPE_SIZE = {
    1: 4,   # FLOAT
    10: 2,  # FLOAT16
    3: 1,   # INT8
    2: 1,   # UINT8
    6: 4,   # INT32
    7: 8,   # INT64
}


def value_info_size_bytes(value_info) -> int:
    """Estimate tensor size in bytes from ONNX ValueInfoProto."""
    t = value_info.type.tensor_type
    if not t.HasField("shape"):
        return 0

    dims = []
    for dim in t.shape.dim:
        if dim.dim_value > 0:
            dims.append(dim.dim_value)
        else:
            return 0  # dynamic/unknown shape in v0.1

    if not dims:
        return 0

    dtype = int(t.elem_type)
    bpe = DTYPE_SIZE.get(dtype, 4)

    return int(np.prod(dims)) * bpe