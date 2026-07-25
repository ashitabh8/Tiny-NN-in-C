"""
Triton GPU code generator - converts IR graph to standalone model.py + weights.npz
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..ir.graph import IRGraph
from ..ir.node import IRNode
from .backend_registry import check_backend
from .memory_planner import (
    assign_buffer_slots,
    calculate_buffer_sizes,
    compute_buffer_last_use,
    node_has_buffer,
)
from .naming import sanitize_name

try:
    from ..profiling.ops.profiling_utils import ProfilingWrapperNode
except ImportError:
    ProfilingWrapperNode = None  # type: ignore


def _ir_dtype_to_torch(dtype: str) -> str:
    if dtype == "int8":
        return "torch.int8"
    if dtype == "int16":
        return "torch.int16"
    return "torch.float32"


class TritonPrinter:
    """Generates Triton-backed Python model from an IR graph."""

    def __init__(self, ir_graph: IRGraph, device: str = "cuda"):
        self.ir_graph = ir_graph
        self.device = device
        self._slot_assignments: Optional[Dict[str, int]] = None

    def generate_all(self, output_dir: str) -> None:
        check_backend("triton")
        os.makedirs(output_dir, exist_ok=True)
        self._write_weights_npz(output_dir)
        model_py = self.generate_model_py()
        with open(os.path.join(output_dir, "model.py"), "w") as f:
            f.write(model_py)
        self._copy_triton_ops(output_dir)

    def _write_weights_npz(self, output_dir: str) -> None:
        arrays = {}
        for name, data in self.ir_graph.parameters.items():
            arrays[sanitize_name(name)] = data
        np.savez(os.path.join(output_dir, "weights.npz"), **arrays)

    def _copy_triton_ops(self, output_dir: str) -> None:
        root = Path(__file__).resolve().parent.parent.parent.parent
        src_dir = root / "src" / "triton_ops"
        dst_dir = os.path.join(output_dir, "triton_ops")
        os.makedirs(dst_dir, exist_ok=True)
        for fname in ("__init__.py", "nn_ops_float.py", "nn_ops_quant.py"):
            src = src_dir / fname
            if src.exists():
                shutil.copy2(src, os.path.join(dst_dir, fname))

    def generate_model_py(self) -> str:
        lines: List[str] = []
        lines.append('"""Auto-generated Triton model. DO NOT EDIT."""')
        lines.append("")
        lines.append("import sys")
        lines.append("from pathlib import Path")
        lines.append("_ROOT = Path(__file__).resolve().parent")
        lines.append("if str(_ROOT) not in sys.path:")
        lines.append("    sys.path.insert(0, str(_ROOT))")
        lines.append("import numpy as np")
        lines.append("import torch")
        lines.append("")
        lines.append("from triton_ops import nn_ops_float as ops_f")
        lines.append("from triton_ops import nn_ops_quant as ops_q")
        lines.append("")
        lines.append(f'_DEVICE = torch.device("{self.device}")')
        lines.append('_WEIGHTS = np.load(_ROOT / "weights.npz")')
        lines.append("")

        for pname in self.ir_graph.parameters:
            sname = sanitize_name(pname)
            arr = self.ir_graph.parameters[pname]
            if arr.dtype == np.int8:
                t = "torch.int8"
            elif arr.dtype == np.int16:
                t = "torch.int16"
            else:
                t = "torch.float32"
            lines.append(
                f'{sname} = torch.from_numpy(_WEIGHTS["{sname}"]).to(_DEVICE, dtype={t}).contiguous()'
            )
        lines.append("")

        buffer_sizes = calculate_buffer_sizes(self.ir_graph)
        order = self.ir_graph.topological_sort()
        last_use = compute_buffer_last_use(order)
        slot_assignments, slot_sizes, slot_dtypes, num_slots = assign_buffer_slots(
            order, buffer_sizes, last_use
        )
        self._slot_assignments = slot_assignments

        for slot_id in range(num_slots):
            td = _ir_dtype_to_torch(slot_dtypes[slot_id])
            lines.append(
                f"slot_{slot_id} = torch.empty({slot_sizes[slot_id]}, "
                f"device=_DEVICE, dtype={td})"
            )
        if num_slots > 0:
            lines.append("")

        lines.append("def model_forward(input: torch.Tensor, output: torch.Tensor) -> None:")
        lines.append('    """Run inference. Input/output: float32 NHWC flat tensors on CUDA."""')
        lines.append("    input = input.contiguous()")
        lines.append("    output = output.contiguous()")

        output_node = self.ir_graph.outputs[0] if self.ir_graph.outputs else None
        indent = "    "
        for node in order:
            if node.op_type in ("input", "method_size"):
                node_code = self._generate_node_code(node)
                for line in node_code:
                    lines.append(indent + line)
                continue
            lines.append(indent + f"# {node.name} [{node.op_type}]")
            node_code = self._generate_node_code(node)
            for line in node_code:
                lines.append(indent + line)
            if output_node is not None and node.name == output_node.name:
                size = buffer_sizes[node.name]
                buf = self._get_buffer_name(node)
                lines.append(indent + f"output.copy_( {buf}[:{size}].view(-1) )")

        self._slot_assignments = None
        lines.append("")
        return "\n".join(lines)

    def _has_nodes_with_dtype(self, dtype: str) -> bool:
        return any(n.dtype == dtype for n in self.ir_graph.nodes)

    def _calculate_buffer_sizes(self) -> Dict[str, int]:
        return calculate_buffer_sizes(self.ir_graph)

    def _generate_node_code(self, node: IRNode) -> List[str]:
        if node.op_type == "input":
            return []
        if hasattr(node, "generate_triton_code"):
            return node.generate_triton_code(self)
        dispatch = {
            "conv2d": self._gen_conv2d,
            "conv1d": self._gen_conv1d,
            "linear": self._gen_linear,
            "relu": self._gen_relu,
            "batchnorm": self._gen_batchnorm,
            "batchnorm1d": self._gen_batchnorm1d,
            "softmax": self._gen_softmax,
            "add": self._gen_add,
            "mul": self._gen_mul,
            "method_mean": self._gen_mean,
            "adaptive_avg_pool": self._gen_adaptive_avg_pool,
            "method_view": self._gen_flatten_or_view,
            "method_flatten": self._gen_flatten_or_view,
            "method_reshape": self._gen_flatten_or_view,
            "method_unsqueeze": self._gen_unsqueeze,
            "method_squeeze": self._gen_squeeze,
            "method_permute": self._gen_permute,
        }
        if node.op_type in dispatch:
            return dispatch[node.op_type](node)
        if node.op_type in ("method_size", "method_getattr", "method_getitem"):
            return []
        if node.op_type == "mul" and node.output_shape is None:
            return []
        return [f"# Unsupported operation: {node.op_type}"]

    def _w(self, name: str) -> str:
        return sanitize_name(name)

    def _sanitize_name(self, name: str) -> str:
        return sanitize_name(name)

    def _get_buffer_name(self, node: IRNode) -> str:
        if node.op_type == "input":
            return "input"
        slots = self._slot_assignments
        if slots is not None:
            if node.op_type == "relu" and node.inputs:
                return f"slot_{slots[node.inputs[0].name]}"
            if node.name in slots:
                return f"slot_{slots[node.name]}"
        return f"buf_{sanitize_name(node.name)}"

    def _get_input_buffer(self, node: IRNode, idx: int) -> str:
        return self._get_buffer_name(node.inputs[idx])

    def _gen_conv2d(self, node: IRNode) -> List[str]:
        inp = self._get_input_buffer(node, 0)
        out = self._get_buffer_name(node)
        w = self._w(node.metadata["weight_name"])
        bias = self._w(node.metadata["bias_name"]) if node.metadata.get("bias_name") else "None"
        k = node.metadata["kernel_size"]
        s = node.metadata["stride"]
        p = node.metadata["padding"]
        ic, oc = node.metadata["in_channels"], node.metadata["out_channels"]
        groups = node.metadata.get("groups", 1)
        kh, kw = (k, k) if isinstance(k, int) else (k[0], k[1])
        sh, sw = (s, s) if isinstance(s, int) else (s[0], s[1])
        ph, pw = (p, p) if isinstance(p, int) else (p[0], p[1])
        in_h, in_w = node.inputs[0].output_shape[2], node.inputs[0].output_shape[3]
        if groups > 1:
            return [
                f"ops_f.depthwise_conv2d_nhwc({inp}, {in_h}, {in_w}, {ic}, {w}, "
                f"{kh}, {kw}, {bias}, {sh}, {sw}, {ph}, {pw}, {out})"
            ]
        return [
            f"ops_f.conv2d_nhwc({inp}, {in_h}, {in_w}, {ic}, {w}, {kh}, {kw}, {oc}, "
            f"{bias}, {sh}, {sw}, {ph}, {pw}, {out})"
        ]

    def _gen_conv1d(self, node: IRNode) -> List[str]:
        inp = self._get_input_buffer(node, 0)
        out = self._get_buffer_name(node)
        w = self._w(node.metadata["weight_name"])
        bias = self._w(node.metadata["bias_name"]) if node.metadata.get("bias_name") else "None"
        kw = int(node.metadata["kernel_size"])
        sw = int(node.metadata["stride"])
        pw = int(node.metadata["padding"])
        ic, oc = node.metadata["in_channels"], node.metadata["out_channels"]
        groups = node.metadata.get("groups", 1)
        in_w = int(node.inputs[0].output_shape[2])
        if groups > 1:
            return [
                f"ops_f.depthwise_conv2d_nhwc({inp}, 1, {in_w}, {ic}, {w}, "
                f"1, {kw}, {bias}, 1, {sw}, 0, {pw}, {out})"
            ]
        return [
            f"ops_f.conv2d_nhwc({inp}, 1, {in_w}, {ic}, {w}, 1, {kw}, {oc}, "
            f"{bias}, 1, {sw}, 0, {pw}, {out})"
        ]

    def _gen_linear(self, node: IRNode) -> List[str]:
        inp = self._get_input_buffer(node, 0)
        out = self._get_buffer_name(node)
        w = self._w(node.metadata["weight_name"])
        bias = self._w(node.metadata["bias_name"]) if node.metadata.get("bias_name") else "None"
        inf, outf = node.metadata["in_features"], node.metadata["out_features"]
        rows = 1
        if node.inputs and node.inputs[0].output_shape:
            shape = list(node.inputs[0].output_shape)
            if shape and shape[0] == 1:
                shape = shape[1:]
            total = 1
            for d in shape:
                total *= d
            if inf > 0 and total % inf == 0:
                rows = total // inf
        if rows == 1:
            return [f"ops_f.dense({inp}, {inf}, {w}, {bias}, {outf}, {out})"]
        return [
            f"for r in range({rows}):",
            f"    ops_f.dense({inp} + r * {inf}, {inf}, {w}, {bias}, {outf}, {out} + r * {outf})",
        ]

    def _gen_relu(self, node: IRNode) -> List[str]:
        buf = self._get_input_buffer(node, 0)
        n = self._calculate_buffer_sizes()[node.name]
        return [f"ops_f.relu({buf}, {n})"]

    def _gen_batchnorm(self, node: IRNode) -> List[str]:
        inp = self._get_input_buffer(node, 0)
        out = self._get_buffer_name(node)
        g, b, m, v = (
            self._w(node.metadata["gamma_name"]),
            self._w(node.metadata["beta_name"]),
            self._w(node.metadata["mean_name"]),
            self._w(node.metadata["var_name"]),
        )
        eps = node.metadata["eps"]
        nf = node.metadata["num_features"]
        h, w = 32, 32
        if node.inputs and node.inputs[0].output_shape and len(node.inputs[0].output_shape) == 4:
            h, w = node.inputs[0].output_shape[2], node.inputs[0].output_shape[3]
        return [f"ops_f.batchnorm2d_nhwc({inp}, {h}, {w}, {nf}, {g}, {b}, {m}, {v}, {eps}, {out})"]

    def _gen_batchnorm1d(self, node: IRNode) -> List[str]:
        inp = self._get_input_buffer(node, 0)
        out = self._get_buffer_name(node)
        g, b, m, v = (
            self._w(node.metadata["gamma_name"]),
            self._w(node.metadata["beta_name"]),
            self._w(node.metadata["mean_name"]),
            self._w(node.metadata["var_name"]),
        )
        eps = node.metadata["eps"]
        nf = node.metadata["num_features"]
        L = int(node.inputs[0].output_shape[2])
        return [f"ops_f.batchnorm2d_nhwc({inp}, 1, {L}, {nf}, {g}, {b}, {m}, {v}, {eps}, {out})"]

    def _gen_softmax(self, node: IRNode) -> List[str]:
        inp = self._get_input_buffer(node, 0)
        out = self._get_buffer_name(node)
        n = self._calculate_buffer_sizes()[node.name]
        return [f"{out}.copy_({inp})", f"ops_f.softmax({out}, {n})"]

    def _gen_add(self, node: IRNode) -> List[str]:
        a, b = self._get_input_buffer(node, 0), self._get_input_buffer(node, 1)
        out = self._get_buffer_name(node)
        n = self._calculate_buffer_sizes()[node.name]
        return [f"ops_f.add_tensors({a}, {b}, {n}, {out})"]

    def _gen_mul(self, node: IRNode) -> List[str]:
        a, b = self._get_input_buffer(node, 0), self._get_input_buffer(node, 1)
        out = self._get_buffer_name(node)
        n = self._calculate_buffer_sizes()[node.name]
        return [f"ops_f.mul_tensors({a}, {b}, {n}, {out})"]

    def _gen_mean(self, node: IRNode) -> List[str]:
        inp = self._get_input_buffer(node, 0)
        out = self._get_buffer_name(node)
        kwargs = node.metadata.get("kwargs", {})
        args = node.metadata.get("args", ())
        dim = kwargs.get("dim")
        if dim is None and args:
            dim = args[0]
        in_node = node.inputs[0]
        shape = list(in_node.output_shape or [])
        if shape and shape[0] == 1:
            shape = shape[1:]
        if isinstance(dim, int) and dim == -1 and len(shape) == 2:
            rows, cols = shape
            if node.dtype == "int8":
                return [
                    f"ops_q.mean_last_dim_int8({inp}, {rows}, {cols}, "
                    f"{node.metadata.get('input_scale', 1.0)}, "
                    f"{node.metadata.get('output_scale', 1.0)}, "
                    f"{node.metadata.get('output_offset', 0)}, {out})"
                ]
            return [f"ops_f.mean_last_dim({inp}, {rows}, {cols}, {out})"]
        if len(shape) == 3:
            c, h, w = shape[0], shape[1], shape[2]
            if node.dtype == "int8":
                return [
                    f"ops_q.mean_hwc_int8({inp}, {h}, {w}, {c}, "
                    f"{node.metadata.get('input_scale', 1.0)}, "
                    f"{node.metadata.get('output_scale', 1.0)}, "
                    f"{node.metadata.get('output_offset', 0)}, {out})"
                ]
            return [f"ops_f.mean_hwc({inp}, {h}, {w}, {c}, {out})"]
        return [f"# TODO mean {node.name}"]

    def _gen_adaptive_avg_pool(self, node: IRNode) -> List[str]:
        inp = self._get_input_buffer(node, 0)
        out = self._get_buffer_name(node)
        h, w, c = 32, 32, 64
        if node.inputs and node.inputs[0].output_shape and len(node.inputs[0].output_shape) == 4:
            _, c, h, w = node.inputs[0].output_shape
        if node.dtype == "int8":
            return [
                f"ops_q.global_average_pool_2d_int8({inp}, {h}, {w}, {c}, "
                f"{node.metadata.get('input_scale', 1.0)}, "
                f"{node.metadata.get('output_scale', 1.0)}, "
                f"{node.metadata.get('output_offset', 0)}, {out})"
            ]
        return [f"ops_f.global_average_pool_2d({inp}, {h}, {w}, {c}, {out})"]

    def _gen_flatten_or_view(self, node: IRNode) -> List[str]:
        inp = self._get_input_buffer(node, 0)
        out = self._get_buffer_name(node)
        n = self._calculate_buffer_sizes()[node.name]
        return [f"{out}.copy_({inp}[:{n}])"]

    def _gen_unsqueeze(self, node: IRNode) -> List[str]:
        return self._gen_flatten_or_view(node)

    def _gen_squeeze(self, node: IRNode) -> List[str]:
        return self._gen_flatten_or_view(node)

    def _gen_permute(self, node: IRNode) -> List[str]:
        inp = self._get_input_buffer(node, 0)
        out = self._get_buffer_name(node)
        raw = list(node.inputs[0].output_shape or [])
        perm_args = list(node.metadata["args"])
        if len(perm_args) == 1 and isinstance(perm_args[0], (tuple, list)):
            perm_args = list(perm_args[0])
        if len(raw) == 4 and raw[0] == 1 and perm_args == [0, 2, 1, 3]:
            c, h, w = raw[1], raw[2], raw[3]
            lines = [
                f"for hh in range({h}):",
                f"    for cc in range({c}):",
                f"        for ww in range({w}):",
                f"            {out}[((hh * {c} + cc) * {w}) + ww] = "
                f"{inp}[((hh * {w} + ww) * {c}) + cc]",
            ]
            return lines
        if raw and raw[0] == 1 and 0 in perm_args:
            dims = raw[1:]
            perm = [int(p) - 1 for p in perm_args if int(p) != 0]
        else:
            dims = raw
            perm = [int(p) for p in perm_args]
        if len(dims) == 3:
            return [
                f"ops_f.permute_3d({inp}, {dims[0]}, {dims[1]}, {dims[2]}, "
                f"{perm[0]}, {perm[1]}, {perm[2]}, {out})"
            ]
        while len(dims) < 4:
            dims.append(1)
        used = set(perm)
        while len(perm) < 4:
            for ax in range(4):
                if ax not in used:
                    perm.append(ax)
                    used.add(ax)
                    break
        return [
            f"ops_f.permute_4d({inp}, {dims[0]}, {dims[1]}, {dims[2]}, {dims[3]}, "
            f"{perm[0]}, {perm[1]}, {perm[2]}, {perm[3]}, {out})"
        ]


def generate_triton_code(ir_graph: IRGraph, output_dir: str, device: str = "cuda") -> None:
    TritonPrinter(ir_graph, device=device).generate_all(output_dir)
