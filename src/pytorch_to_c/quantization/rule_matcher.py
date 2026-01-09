"""
Rule Matcher - Matches quantization rules to IR nodes
"""

from typing import List, Optional
from ..ir.node import IRNode
from .rules import QuantRule


class RuleMatcher:
    """
    Matches quantization rules to IR nodes.
    
    Rules are checked in order - first match wins.
    """
    
    def __init__(self, rules: List[QuantRule]):
        """
        Initialize the rule matcher.
        
        Args:
            rules: List of quantization rules (checked in order)
        """
        self.rules = rules
    
    def find_matching_rule(self, node: IRNode) -> Optional[QuantRule]:
        """
        Find the first rule that matches a node.
        
        Args:
            node: The IRNode to match
            
        Returns:
            The matching QuantRule, or None if no match
        """
        for rule in self.rules:
            if rule.matches(node):
                return rule
        return None
    
    def get_quantizable_nodes(self, ir_graph) -> List[IRNode]:
        """
        Get all nodes that have matching rules.
        
        Args:
            ir_graph: The IRGraph to search
            
        Returns:
            List of nodes that match at least one rule
        """
        quantizable = []
        for node in ir_graph.nodes:
            if self.find_matching_rule(node) is not None:
                quantizable.append(node)
        return quantizable
    
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

