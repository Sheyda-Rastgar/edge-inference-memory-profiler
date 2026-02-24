from __future__ import annotations

from typing import List, Tuple, Dict

from .onnx_utils import value_info_size_bytes
from .lifetime_analyzer import build_tensor_lifetimes
from .reuse_allocator import greedy_reuse_peak


def _collect_activation_tensors(model) -> List[Tuple[str, int]]:
    initializer_names = {init.name for init in model.graph.initializer}
    activations: List[Tuple[str, int]] = []

    def add_if_activation(vi):
        if vi.name in initializer_names:
            return
        size = value_info_size_bytes(vi)
        if size > 0:
            activations.append((vi.name, size))

    # Intermediates
    for vi in model.graph.value_info:
        add_if_activation(vi)

    # Inputs / outputs
    for vi in model.graph.input:
        add_if_activation(vi)
    for vi in model.graph.output:
        add_if_activation(vi)

    # De-dup
    seen = set()
    unique = []
    for name, size in activations:
        if name not in seen:
            unique.append((name, size))
            seen.add(name)

    return unique


def _estimate_weight_bytes(model) -> int:
    weight_bytes = 0
    for init in model.graph.initializer:
        element_count = 1
        for d in init.dims:
            element_count *= int(d)
        dtype = int(init.data_type)
        bpe = {1: 4, 10: 2, 3: 1, 2: 1, 6: 4, 7: 8}.get(dtype, 4)
        weight_bytes += element_count * bpe
    return weight_bytes


def compute_memory_report(model):
    """
    Returns:
      weight_bytes
      naive_peak_bytes
      reuse_peak_bytes
      activations (list)
      allocations (dict)  # tensor -> (buf_id, buf_size)
      meta (dict)         # extra info (input_bytes, reduction_pct, etc.)
    """
    activations = _collect_activation_tensors(model)
    naive_peak_bytes = sum(size for _, size in activations)
    weight_bytes = _estimate_weight_bytes(model)

    lifetimes = build_tensor_lifetimes(model)
    reuse_peak_bytes, allocations = greedy_reuse_peak(activations, lifetimes)

    # Inputs have no producer node -> not included in lifetimes-based allocation.
    input_names = {vi.name for vi in model.graph.input}
    input_bytes = sum(size for name, size in activations if name in input_names)

    # Add input memory as an always-live baseline in v0.2
    reuse_peak_bytes_with_inputs = reuse_peak_bytes + input_bytes

    reduction_pct = 0.0
    if naive_peak_bytes > 0:
        reduction_pct = (naive_peak_bytes - reuse_peak_bytes_with_inputs) / naive_peak_bytes * 100.0

    meta = {
        "input_bytes": input_bytes,
        "reuse_peak_bytes_raw": reuse_peak_bytes,  # without input baseline
        "reuse_peak_bytes_with_inputs": reuse_peak_bytes_with_inputs,
        "reduction_pct": reduction_pct,
        "num_activation_tensors": len(activations),
        "num_reused_tensors": len([n for (n, _) in activations if n in lifetimes]),
    }

    return (
        weight_bytes,
        naive_peak_bytes,
        reuse_peak_bytes_with_inputs,
        activations,
        allocations,
        meta,
    )