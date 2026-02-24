from __future__ import annotations

from typing import Dict, List, Tuple


def greedy_reuse_peak(
    activations: List[Tuple[str, int]],
    lifetimes: Dict[str, Tuple[int, int]],
) -> Tuple[int, Dict[str, Tuple[int, int]]]:
    """
    Greedy buffer reuse simulation.

    activations: [(tensor_name, size_bytes)]  (activation tensors only)
    lifetimes: {tensor_name: (start_node_idx, end_node_idx)}

    Returns:
      peak_bytes: peak memory with reuse
      allocations: {tensor_name: (buffer_id, buffer_size_bytes)}
    """
    act = [(name, size) for (name, size) in activations if name in lifetimes]
    act.sort(key=lambda x: lifetimes[x[0]][0])  # sort by start time

    # Active buffers: [(end_idx, buf_id, buf_size)]
    active: List[Tuple[int, int, int]] = []
    # Free buffers: [(buf_id, buf_size)]
    free_buffers: List[Tuple[int, int]] = []

    allocations: Dict[str, Tuple[int, int]] = {}

    next_buf_id = 0
    current_bytes = 0
    peak_bytes = 0

    for name, size in act:
        start, end = lifetimes[name]

        # Free buffers whose lifetimes ended before current start
        still_active: List[Tuple[int, int, int]] = []
        for end_idx, buf_id, buf_size in active:
            if end_idx < start:
                current_bytes -= buf_size
                free_buffers.append((buf_id, buf_size))
            else:
                still_active.append((end_idx, buf_id, buf_size))
        active = still_active

        # Find first-fit reusable buffer
        chosen_id = None
        chosen_size = None
        chosen_idx = None
        for i, (buf_id, buf_size) in enumerate(free_buffers):
            if buf_size >= size:
                chosen_id = buf_id
                chosen_size = buf_size
                chosen_idx = i
                break

        if chosen_id is not None:
            free_buffers.pop(chosen_idx)  # remove reused buffer
            buf_id, buf_size = chosen_id, chosen_size
        else:
            buf_id, buf_size = next_buf_id, size
            next_buf_id += 1

        allocations[name] = (buf_id, buf_size)
        active.append((end, buf_id, buf_size))
        active.sort(key=lambda x: x[0])

        current_bytes += buf_size
        if current_bytes > peak_bytes:
            peak_bytes = current_bytes

    return peak_bytes, allocations