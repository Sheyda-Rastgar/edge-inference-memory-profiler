from .onnx_utils import value_info_size_bytes


def compute_naive_peak(model):
    """
    Naive peak activation memory (no reuse).
    Excludes initializers (weights) from activations.
    """
    initializer_names = {init.name for init in model.graph.initializer}

    activations = []

    def add_if_activation(vi):
        if vi.name in initializer_names:
            return
        size = value_info_size_bytes(vi)
        if size > 0:
            activations.append((vi.name, size))

    # Intermediate tensors (shape-inferred)
    for vi in model.graph.value_info:
        add_if_activation(vi)

    # Inputs/outputs may not be in value_info
    for vi in model.graph.input:
        add_if_activation(vi)
    for vi in model.graph.output:
        add_if_activation(vi)

    # De-dup by name
    seen = set()
    unique = []
    for name, size in activations:
        if name not in seen:
            unique.append((name, size))
            seen.add(name)

    peak_bytes = sum(size for _, size in unique)

    # Weights separately (initializers)
    weight_bytes = 0
    for init in model.graph.initializer:
        element_count = 1
        for d in init.dims:
            element_count *= int(d)
        dtype = int(init.data_type)
        bpe = {1: 4, 10: 2, 3: 1, 2: 1, 6: 4, 7: 8}.get(dtype, 4)
        weight_bytes += element_count * bpe

    return peak_bytes, unique, weight_bytes