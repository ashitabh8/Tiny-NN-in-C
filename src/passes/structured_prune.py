"""
Structured pruning pass — removes weakest output channels/features from linear/conv layers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np

from src.passes.base import IRPass
from src.pytorch_to_c.ir.graph import IRGraph
from src.pytorch_to_c.ir.node import IRNode


@dataclass
class PruneRule:
    pattern: str
    amount: float
    criterion: Literal["l2", "activation"] = "l2"

    def __post_init__(self):
        self._compiled = re.compile(self.pattern)

    def matches(self, node: IRNode) -> bool:
        return self._compiled.fullmatch(node.name) is not None


class StructuredPruningPass(IRPass):
    """Prune output channels/columns and propagate shape changes downstream."""

    def __init__(
        self,
        rules: List[PruneRule],
        activation_stats: Optional[Dict] = None,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.rules = rules
        self.activation_stats = activation_stats or {}

    def apply(self, ir_graph: IRGraph) -> IRGraph:
        pruned_layers: Dict[str, np.ndarray] = {}
        self.stats = {"pruned": []}

        for node in list(ir_graph.nodes):
            rule = self._match_rule(node)
            if rule is None:
                continue
            if node in ir_graph.outputs:
                self._log(f"skip output layer {node.name}")
                continue
            if node.op_type == "linear":
                keep = self._prune_linear(ir_graph, node, rule, pruned_layers)
            elif node.op_type == "conv2d":
                keep = self._prune_conv(ir_graph, node, rule, pruned_layers)
            else:
                continue
            if keep is not None:
                self.stats["pruned"].append({"name": node.name, "keep": int(keep.size)})

        ir_graph.rebuild_node_map()
        ir_graph.validate()
        return ir_graph

    def _match_rule(self, node: IRNode) -> Optional[PruneRule]:
        for rule in self.rules:
            if rule.matches(node):
                return rule
        return None

    def _importance(self, node: IRNode, weights: np.ndarray, rule: PruneRule) -> np.ndarray:
        if rule.criterion == "activation" and node.name in self.activation_stats:
            stats = self.activation_stats[node.name]
            if hasattr(stats, "absmax"):
                return np.full(weights.shape[-1], stats.absmax, dtype=np.float64)
        if weights.ndim == 2:
            return np.linalg.norm(weights, axis=0)
        return np.linalg.norm(weights.reshape(-1, weights.shape[-1]), axis=0)

    def _prune_linear(
        self,
        ir_graph: IRGraph,
        node: IRNode,
        rule: PruneRule,
        pruned_layers: Dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        wn = node.metadata.get("weight_name")
        if not wn or wn not in ir_graph.parameters:
            return None
        W = ir_graph.parameters[wn]
        out_features = W.shape[1]
        n_prune = max(1, int(out_features * rule.amount))
        n_keep = max(1, out_features - n_prune)
        scores = self._importance(node, W, rule)
        keep = np.argsort(scores)[-n_keep:]

        ir_graph.parameters[wn] = W[:, keep]
        bn = node.metadata.get("bias_name")
        if bn and bn in ir_graph.parameters:
            ir_graph.parameters[bn] = ir_graph.parameters[bn][keep]
        node.metadata["out_features"] = n_keep
        if node.output_shape:
            sh = list(node.output_shape)
            sh[-1] = n_keep
            node.output_shape = tuple(sh)

        pruned_layers[node.name] = keep
        self._propagate_linear_consumers(ir_graph, node, keep)
        return keep

    def _propagate_linear_consumers(
        self, ir_graph: IRGraph, producer: IRNode, keep: np.ndarray
    ) -> None:
        for user in producer.users:
            if user.op_type != "linear":
                if user.op_type in ("add", "mul"):
                    raise ValueError(
                        f"Structured pruning at '{producer.name}' hits '{user.name}' "
                        f"({user.op_type}); residual branches not supported in v1."
                    )
                continue
            wn = user.metadata.get("weight_name")
            if not wn or wn not in ir_graph.parameters:
                continue
            W = ir_graph.parameters[wn]
            ir_graph.parameters[wn] = W[keep, :]
            user.metadata["in_features"] = len(keep)
            if user.output_shape and len(user.output_shape) >= 1:
                pass

    def _prune_conv(
        self,
        ir_graph: IRGraph,
        node: IRNode,
        rule: PruneRule,
        pruned_layers: Dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        wn = node.metadata.get("weight_name")
        if not wn or wn not in ir_graph.parameters:
            return None
        W = ir_graph.parameters[wn]
        out_c = W.shape[-1]
        n_prune = max(1, int(out_c * rule.amount))
        n_keep = max(1, out_c - n_prune)
        scores = self._importance(node, W, rule)
        keep = np.argsort(scores)[-n_keep:]

        if W.ndim == 4:
            ir_graph.parameters[wn] = W[..., keep]
        else:
            ir_graph.parameters[wn] = W[..., keep]
        bn = node.metadata.get("bias_name")
        if bn and bn in ir_graph.parameters:
            ir_graph.parameters[bn] = ir_graph.parameters[bn][keep]
        node.metadata["out_channels"] = n_keep
        if node.output_shape and len(node.output_shape) == 4:
            sh = list(node.output_shape)
            sh[1] = n_keep
            node.output_shape = tuple(sh)

        pruned_layers[node.name] = keep
        self._propagate_conv_consumers(ir_graph, node, keep, n_keep)
        return keep

    def _propagate_conv_consumers(
        self, ir_graph: IRGraph, producer: IRNode, keep: np.ndarray, n_keep: int
    ) -> None:
        for user in producer.users:
            if user.op_type == "conv2d":
                wn = user.metadata.get("weight_name")
                if wn and wn in ir_graph.parameters:
                    W = ir_graph.parameters[wn]
                    if W.ndim == 4:
                        ir_graph.parameters[wn] = W[:, :, keep, :]
                    user.metadata["in_channels"] = n_keep
            elif user.op_type == "linear":
                wn = user.metadata.get("weight_name")
                if wn and wn in ir_graph.parameters:
                    W = ir_graph.parameters[wn]
                    ir_graph.parameters[wn] = W[keep, :]
                    user.metadata["in_features"] = n_keep
