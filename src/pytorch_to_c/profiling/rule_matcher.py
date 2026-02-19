"""
Rule Matcher - Matches profiling rules to IR nodes
"""

from typing import List, Optional

from ..ir.node import IRNode
from .rules import ProfilingRule


class ProfilingRuleMatcher:
    """
    Matches profiling rules to IR nodes.
    Rules are checked in order - first match wins.
    """

    def __init__(self, rules: List[ProfilingRule]):
        """
        Initialize the rule matcher.

        Args:
            rules: List of profiling rules (checked in order)
        """
        self.rules = rules

    def find_matching_rule(self, node: IRNode) -> Optional[ProfilingRule]:
        """
        Find the first rule that matches a node.

        Args:
            node: The IRNode to match

        Returns:
            The matching ProfilingRule, or None if no match
        """
        for rule in self.rules:
            if rule.matches(node):
                return rule
        return None

    def get_node_rule_mapping(self, ir_graph) -> dict:
        """
        Get a mapping of nodes to their matching rules.

        Args:
            ir_graph: The IRGraph to search

        Returns:
            Dictionary mapping node -> rule
        """
        mapping = {}
        for node in ir_graph.nodes:
            rule = self.find_matching_rule(node)
            if rule is not None:
                mapping[node] = rule
        return mapping
