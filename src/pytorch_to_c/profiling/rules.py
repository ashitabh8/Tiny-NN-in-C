"""
Profiling Rules - Define which float nodes to wrap with timing/label
"""

import re
from typing import Optional

from ..ir.node import IRNode


class ProfilingRule:
    """
    Rule to replace a float node with a ProfilingWrapperNode.
    Matches by regex on node name; optional label string (printed at node start).
    """

    def __init__(self, pattern: str, label: Optional[str] = None):
        """
        Initialize a profiling rule.

        Args:
            pattern: Regex pattern to match node names
            label: String printed at the start of the node (if None, use node.name)
        """
        self.pattern = pattern
        self.label = label
        self._compiled_pattern = re.compile(pattern)

    def matches(self, node: IRNode) -> bool:
        """
        Check if this rule applies to a node.

        Args:
            node: IRNode to check

        Returns:
            True if the node name matches the pattern
        """
        return self._compiled_pattern.match(node.name) is not None

    def create_profiling_node(self, node: IRNode) -> "IRNode":
        """
        Create a ProfilingWrapperNode that wraps the given float node.

        Args:
            node: The float IRNode to wrap

        Returns:
            ProfilingWrapperNode instance
        """
        from .ops.profiling_utils import ProfilingWrapperNode
        return ProfilingWrapperNode(original_node=node, label=self.label or node.name)
