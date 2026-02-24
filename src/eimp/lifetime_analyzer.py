from __future__ import annotations

from typing import Dict, Tuple


def build_tensor_lifetimes(model) -> Dict[str, Tuple[int, int]]:
    """
    Build lifetimes for tensors produced by nodes.
    Lifetime: (producer_node_index, last_consumer_node_index)

    Notes:
    - Only tensors produced by graph nodes get lifetimes.
    - Graph inputs have no producer node; we treat them separately in v0.2.
    """
    producer: Dict[str, int] = {}
    last_consumer: Dict[str, int] = {}

    for node_idx, node in enumerate(model.graph.node):
        for out_name in node.output:
            if out_name and out_name not in producer:
                producer[out_name] = node_idx

        for in_name in node.input:
            if in_name:
                last_consumer[in_name] = node_idx

    lifetimes: Dict[str, Tuple[int, int]] = {}
    for tensor_name, start in producer.items():
        end = last_consumer.get(tensor_name, start)
        lifetimes[tensor_name] = (start, end)

    return lifetimes