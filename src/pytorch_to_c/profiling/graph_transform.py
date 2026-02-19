"""
Graph Transformation - Apply profiling rules to IR graph
"""

from typing import Dict

from ..ir.graph import IRGraph
from ..ir.node import IRNode
from .rules import ProfilingRule
from .rule_matcher import ProfilingRuleMatcher


class ProfilingTransform:
    """
    Transform float IR graph by replacing matched nodes with ProfilingWrapperNodes.
    No pre/post node insertion; no weight changes. All nodes remain float32.
    """

    def __init__(self, rules: list):
        """
        Initialize the profiling transform.

        Args:
            rules: List of ProfilingRule to apply
        """
        self.rules = rules
        self.matcher = ProfilingRuleMatcher(rules)

    def apply(self, ir_graph: IRGraph) -> IRGraph:
        """
        Apply profiling rules to the IR graph.

        Args:
            ir_graph: The float IR graph to transform

        Returns:
            Modified IRGraph with profiling wrapper nodes
        """
        nodes_to_profile = self.matcher.get_node_rule_mapping(ir_graph)

        for node, rule in nodes_to_profile.items():
            profiling_node = rule.create_profiling_node(node)
            self._replace_node(ir_graph, node, profiling_node)

        return ir_graph

    def _replace_node(self, ir_graph: IRGraph, old_node: IRNode, new_node: IRNode):
        """
        Replace a node in the graph while maintaining connections.

        Args:
            ir_graph: The IR graph
            old_node: Node to replace
            new_node: New node
        """
        idx = ir_graph.nodes.index(old_node)
        ir_graph.nodes[idx] = new_node

        for i, out_node in enumerate(ir_graph.outputs):
            if out_node is old_node:
                ir_graph.outputs[i] = new_node

        for user in old_node.users:
            for i, inp in enumerate(user.inputs):
                if inp is old_node:
                    user.inputs[i] = new_node

        new_node.users = old_node.users.copy()

        for inp in old_node.inputs:
            for i, user in enumerate(inp.users):
                if user is old_node:
                    inp.users[i] = new_node
