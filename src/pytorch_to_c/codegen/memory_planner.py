"""
Backend-agnostic activation buffer planning for code generators.

Pure functions over IRGraph/IRNode — no C or Triton assumptions.
Slot assignment groups buffers by IR dtype (float32, int8, int16).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ..ir.graph import IRGraph
from ..ir.node import IRNode
from .parallel_regions import find_parallel_regions, region_index_spans

# Ops that produce no tensor buffer (shape metadata only).
_SHAPE_ONLY_OPS = frozenset({"method_size", "method_getattr", "method_getitem"})


def node_has_buffer(node: IRNode) -> bool:
    """True if this node produces an output buffer (not input or shape-only ops)."""
    if node.op_type in _SHAPE_ONLY_OPS:
        return False
    if node.op_type == "mul" and node.output_shape is None:
        return False
    return node.op_type != "input"


def _node_buffer_size_from_shape(sizes: Dict[str, int], node: IRNode) -> Optional[int]:
    """Return buffer size for a node from sizes dict or output_shape. None if unknown."""
    if node.name in sizes:
        return sizes[node.name]
    if node.op_type == "input" and node.output_shape is not None:
        shape = node.output_shape
        if len(shape) > 0 and shape[0] == 1:
            shape = shape[1:]
        return math.prod(shape) if shape else 1
    if node.output_shape is not None:
        shape = node.output_shape
        if len(shape) > 0 and shape[0] == 1:
            shape = shape[1:]
        return math.prod(shape) if shape else 1
    return None


def calculate_buffer_sizes(ir_graph: IRGraph) -> Dict[str, int]:
    """
    Calculate buffer sizes for each node using inferred shapes.

    Returns:
        Dictionary mapping node name to buffer size (total number of elements).
    """
    sizes: Dict[str, int] = {}

    for node in ir_graph.nodes:
        if node.op_type == "input":
            continue
        if not node_has_buffer(node):
            continue

        if node.output_shape is not None:
            shape = node.output_shape
            if len(shape) > 0 and shape[0] == 1:
                shape = shape[1:]
            sizes[node.name] = math.prod(shape) if shape else 1
            continue

        if node.op_type == "linear":
            if "out_features" not in node.metadata:
                raise ValueError(
                    f"{node.name} (linear): missing metadata 'out_features'; "
                    f"need shape inference or metadata"
                )
            sizes[node.name] = node.metadata["out_features"]
        elif node.op_type in ("conv2d", "conv1d"):
            raise ValueError(
                f"{node.name} ({node.op_type}): missing output_shape; "
                f"run with example_input for shape inference"
            )
        elif node.op_type in ("relu", "gelu", "softmax", "batchnorm", "batchnorm1d"):
            if not node.inputs:
                raise ValueError(f"{node.name} ({node.op_type}): no input node")
            input_size = _node_buffer_size_from_shape(sizes, node.inputs[0])
            if input_size is None:
                raise ValueError(
                    f"{node.name} ({node.op_type}): input shape unknown; "
                    f"run with example_input for shape inference"
                )
            sizes[node.name] = input_size
        elif node.op_type == "adaptive_avg_pool":
            if (
                not node.inputs
                or not node.inputs[0].output_shape
                or len(node.inputs[0].output_shape) != 4
            ):
                raise ValueError(
                    f"{node.name} (adaptive_avg_pool): need input with 4D shape [B,C,H,W]"
                )
            sizes[node.name] = node.inputs[0].output_shape[1]
        elif node.op_type in (
            "method_view",
            "method_flatten",
            "method_reshape",
            "method_unsqueeze",
            "method_squeeze",
            "method_permute",
        ):
            if not node.inputs:
                raise ValueError(f"{node.name} ({node.op_type}): no input node")
            input_size = _node_buffer_size_from_shape(sizes, node.inputs[0])
            if input_size is None:
                raise ValueError(
                    f"{node.name} ({node.op_type}): input shape unknown; "
                    f"run with example_input for shape inference"
                )
            sizes[node.name] = input_size
        elif node.op_type in ("method_getattr", "method_getitem"):
            continue
        elif node.op_type == "mul":
            if node.output_shape is None:
                continue
            if not node.inputs:
                raise ValueError(f"{node.name} (mul): no input node")
            input_size = _node_buffer_size_from_shape(sizes, node.inputs[0])
            if input_size is None:
                raise ValueError(f"{node.name} (mul): input shape unknown")
            sizes[node.name] = input_size
        else:
            raise ValueError(
                f"{node.name}: unknown op_type '{node.op_type}' and no output_shape; "
                f"run with example_input for shape inference"
            )

    return sizes


def compute_buffer_last_use(order: List[IRNode]) -> Dict[str, IRNode]:
    """
    For each buffer-producing node, find the last node (in execution order) that uses it.
    """
    last_use: Dict[str, IRNode] = {}
    for node in order:
        for inp in node.inputs:
            last_use[inp.name] = node
    return last_use


def assign_buffer_slots(
    order: List[IRNode],
    buffer_sizes: Dict[str, int],
    last_use: Dict[str, IRNode],
) -> Tuple[Dict[str, int], Dict[int, int], Dict[int, str], int]:
    """
    Assign each buffer-producing node to a reusable slot using interval graph coloring.

    Relu nodes share their input's slot (in-place). Dtype grouping uses IR-level dtype.

    Returns:
        (slot_assignments, slot_sizes, slot_dtypes, num_slots)
    """
    order_index = {n.name: i for i, n in enumerate(order)}

    last_use_idx: Dict[str, int] = {}
    for node in order:
        if node.op_type in ("input", "method_size"):
            continue
        if node.op_type in ("relu", "gelu"):
            continue
        if not node_has_buffer(node):
            continue
        def_idx = order_index[node.name]
        lu_node = last_use.get(node.name, node)
        last_use_idx[node.name] = order_index[lu_node.name]

    for node in order:
        if node.op_type in ("relu", "gelu") and node.inputs:
            inp = node.inputs[0]
            if inp.name not in last_use_idx:
                continue
            lu_relu = last_use.get(node.name, node)
            relu_last_idx = order_index[lu_relu.name]
            last_use_idx[inp.name] = max(last_use_idx[inp.name], relu_last_idx)

    # Parallel regions (e.g. LQER diamonds): the two chains execute
    # concurrently (OpenMP sections in the C backend), so their relative
    # order is undefined. Every buffer defined or consumed inside a region
    # must therefore stay live until the join node executes — otherwise
    # interval coloring could alias a buffer from one chain with a buffer
    # from the other.
    region_spans = region_index_spans(order, find_parallel_regions(order))
    if region_spans:
        for node in order:
            join_idx = region_spans.get(node.name)
            if join_idx is None:
                continue
            if node.name in last_use_idx:
                last_use_idx[node.name] = max(last_use_idx[node.name], join_idx)
            for inp in node.inputs:
                if inp.name in last_use_idx:
                    last_use_idx[inp.name] = max(last_use_idx[inp.name], join_idx)

    intervals: List[Tuple[str, int, int, int, str]] = []
    for node in order:
        if node.op_type in ("input", "method_size", "relu", "gelu"):
            continue
        if not node_has_buffer(node):
            continue
        def_idx = order_index[node.name]
        lu_idx = last_use_idx[node.name]
        size = buffer_sizes.get(node.name, 1024)
        intervals.append((node.name, def_idx, lu_idx, size, node.dtype))

    intervals.sort(key=lambda x: x[1])

    slot_assignments: Dict[str, int] = {}
    slot_last_use: Dict[int, int] = {}
    slot_sizes: Dict[int, int] = {}
    slot_dtypes: Dict[int, str] = {}

    for node_name, def_idx, lu_idx, size, dtype in intervals:
        found_slot = None
        for slot_id in sorted(slot_last_use.keys()):
            if slot_last_use[slot_id] < def_idx and slot_dtypes[slot_id] == dtype:
                found_slot = slot_id
                break
        if found_slot is None:
            found_slot = len(slot_last_use)
            slot_last_use[found_slot] = -1
            slot_dtypes[found_slot] = dtype
        slot_assignments[node_name] = found_slot
        slot_last_use[found_slot] = lu_idx
        slot_sizes[found_slot] = max(slot_sizes.get(found_slot, 0), size)

    num_slots = len(slot_sizes)
    return slot_assignments, slot_sizes, slot_dtypes, num_slots
