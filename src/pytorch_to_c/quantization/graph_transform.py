"""
Graph Transformation - Apply quantization rules to IR graph
"""

from typing import List, Dict, Optional
from ..ir.graph import IRGraph
from ..ir.node import IRNode
from ..ir.quant_node import QuantIRNode
from .rules import QuantRule, DynamicQuantRuleMinMaxPerTensor
from .rule_matcher import RuleMatcher
from .ops.quant_utils import QuantizeNode, DequantizeNode


class QuantizationTransform:
    """
    Transform float IR graph to quantized IR graph.
    
    Process:
    1. Find nodes matching rules
    2. Replace with QuantIRNodes
    3. Insert pre/post nodes as specified by each QuantIRNode (node-controlled)
    4. Quantize weights during compilation
    5. Validate dtype compatibility
    
    NOTE: Pre/post node insertion is controlled by the QuantIRNode itself via
    get_pre_nodes() and get_post_nodes(). This allows each node type to decide
    what conversions it needs (static vs dynamic quantization, etc.)
    """
    
    def __init__(self, rules: List[QuantRule]):
        """
        Initialize the quantization transform.
        
        Args:
            rules: List of quantization rules to apply
        """
        self.rules = rules
        self.matcher = RuleMatcher(rules)
    
    def apply(self, ir_graph: IRGraph) -> IRGraph:
        """
        Apply quantization rules to IR graph.
        
        Args:
            ir_graph: The float IR graph to transform
            
        Returns:
            Modified IRGraph with quantized nodes
        """
        # Step 1: Find nodes to quantize and their rules
        nodes_to_quantize = self._find_nodes_to_quantize(ir_graph)
        
        # Step 2: Replace nodes with quantized versions
        self._replace_nodes(ir_graph, nodes_to_quantize)
        
        # Step 3: Insert pre/post nodes as specified by each QuantIRNode
        self._insert_node_controlled_conversions(ir_graph)
        
        # Step 4: Ensure final output is float32 (for C API compatibility)
        self._validate_float_output(ir_graph)
        
        # Step 5: Quantize weights during compilation
        self._quantize_weights(ir_graph, nodes_to_quantize)
        
        # Step 6: Validate dtype compatibility
        self._validate_graph(ir_graph)
        
        return ir_graph
    
    def _find_nodes_to_quantize(self, ir_graph: IRGraph) -> Dict[IRNode, QuantRule]:
        """
        Find all nodes that should be quantized based on rules.
        
        Args:
            ir_graph: The IR graph
            
        Returns:
            Dictionary mapping node -> rule
        """
        nodes_to_quantize = {}
        
        for node in ir_graph.nodes:
            rule = self.matcher.find_matching_rule(node)
            if rule is not None:
                nodes_to_quantize[node] = rule
        
        return nodes_to_quantize
    
    def _replace_nodes(self, ir_graph: IRGraph, nodes_to_quantize: Dict[IRNode, QuantRule]):
        """
        Replace float nodes with quantized versions.
        
        Args:
            ir_graph: The IR graph
            nodes_to_quantize: Dictionary mapping node -> rule
        """
        for node, rule in nodes_to_quantize.items():
            try:
                # For dynamic rules, need to provide weights
                if isinstance(rule, DynamicQuantRuleMinMaxPerTensor):
                    weight_name = node.metadata.get('weight_name')
                    if weight_name and weight_name in ir_graph.parameters:
                        weights = ir_graph.parameters[weight_name]
                        quant_node = rule.create_quant_node_with_weights(node, weights)
                    else:
                        raise ValueError(f"Cannot find weights for node {node.name}")
                else:
                    quant_node = rule.create_quant_node(node)
                
                self._replace_node(ir_graph, node, quant_node)
                
            except ValueError as e:
                # Fail fast (design decision)
                raise ValueError(
                    f"Failed to quantize node '{node.name}' ({node.op_type}): {e}"
                )
    
    def _replace_node(self, ir_graph: IRGraph, old_node: IRNode, new_node: IRNode):
        """
        Replace a node in the graph while maintaining connections.
        
        Args:
            ir_graph: The IR graph
            old_node: Node to replace
            new_node: New node
        """
        # Update node list
        idx = ir_graph.nodes.index(old_node)
        ir_graph.nodes[idx] = new_node
        
        # Update outputs list if this was an output node
        for i, out_node in enumerate(ir_graph.outputs):
            if out_node is old_node:
                ir_graph.outputs[i] = new_node
        
        # Update users to point to new node
        for user in old_node.users:
            for i, inp in enumerate(user.inputs):
                if inp is old_node:
                    user.inputs[i] = new_node
        
        # Copy users to new node
        new_node.users = old_node.users.copy()
        
        # Update input nodes' users list
        for inp in old_node.inputs:
            for i, user in enumerate(inp.users):
                if user is old_node:
                    inp.users[i] = new_node
    
    def _insert_node_controlled_conversions(self, ir_graph: IRGraph):
        """
        Insert pre/post nodes as specified by each QuantIRNode.
        
        This is the node-controlled approach: each QuantIRNode decides what
        conversion nodes it needs via get_pre_nodes() and get_post_nodes().
        
        For dynamic quantization: DynamicQuantizeInputNode (computes scale at runtime)
        For static quantization: QuantizeNode (uses pre-calibrated scale)
        """
        # Collect all insertions first (to avoid modifying list while iterating)
        insertions = []  # List of (position, node, type: 'pre'|'post')
        
        for node in ir_graph.nodes:
            if isinstance(node, QuantIRNode):
                # Get pre/post nodes from the node itself
                pre_nodes = node.get_pre_nodes()
                post_nodes = node.get_post_nodes()
                
                # Always add pre nodes if the QuantNode requests them
                # The QuantNode knows what it needs - don't second-guess it
                # (e.g., consecutive quant layers: fc1 -> dequant -> quant -> fc2)
                if pre_nodes and node.inputs:
                    insertions.append((node, pre_nodes, 'pre'))
                
                # Always add post nodes (to return to float32 for flexibility)
                if post_nodes:
                    insertions.append((node, post_nodes, 'post'))
        
        # Now perform insertions
        for quant_node, nodes_to_add, insert_type in insertions:
            if insert_type == 'pre':
                self._insert_pre_nodes(ir_graph, quant_node, nodes_to_add)
            else:
                self._insert_post_nodes(ir_graph, quant_node, nodes_to_add)
    
    def _insert_pre_nodes(self, ir_graph: IRGraph, target_node: IRNode, pre_nodes: List[IRNode]):
        """
        Insert pre-nodes before a target node.
        
        Chain: input -> pre_node1 -> pre_node2 -> ... -> target_node
        """
        if not pre_nodes or not target_node.inputs:
            return
        
        # Get the original input to target_node
        original_input = target_node.inputs[0]
        
        # Find position to insert (before target_node)
        target_idx = ir_graph.nodes.index(target_node)
        
        # Chain pre_nodes: input -> pre_node1 -> pre_node2 -> ... -> target
        prev_node = original_input
        insert_offset = 0
        
        for pre_node in pre_nodes:
            # Set output_shape from the input node (pre nodes pass through same shape)
            if pre_node.output_shape is None and prev_node.output_shape is not None:
                pre_node.output_shape = prev_node.output_shape
            
            # Set up connections
            pre_node.inputs = [prev_node]
            pre_node.users = []
            
            # Update prev_node's users
            if target_node in prev_node.users:
                prev_node.users.remove(target_node)
            prev_node.users.append(pre_node)
            
            # Insert into graph
            ir_graph.nodes.insert(target_idx + insert_offset, pre_node)
            insert_offset += 1
            
            prev_node = pre_node
        
        # Connect last pre_node to target_node
        target_node.inputs[0] = prev_node
        prev_node.users.append(target_node)
    
    def _insert_post_nodes(self, ir_graph: IRGraph, source_node: IRNode, post_nodes: List[IRNode]):
        """
        Insert post-nodes after a source node.
        
        Chain: source_node -> post_node1 -> post_node2 -> ... -> users
        """
        if not post_nodes:
            return
        
        # Get the original users of source_node
        original_users = source_node.users.copy()
        
        # Find position to insert (after source_node)
        source_idx = ir_graph.nodes.index(source_node)
        
        # Chain post_nodes: source -> post_node1 -> post_node2 -> ... -> users
        prev_node = source_node
        prev_node.users = []  # Will be updated
        
        for i, post_node in enumerate(post_nodes):
            # Set output_shape from the source node (post nodes pass through same shape)
            if post_node.output_shape is None and source_node.output_shape is not None:
                post_node.output_shape = source_node.output_shape
            
            # Set up connections
            post_node.inputs = [prev_node]
            post_node.users = []
            
            # Update prev_node's users
            prev_node.users.append(post_node)
            
            # Insert into graph
            ir_graph.nodes.insert(source_idx + 1 + i, post_node)
            
            prev_node = post_node
        
        # Connect last post_node to original users
        last_post_node = prev_node
        last_post_node.users = []
        
        for user in original_users:
            # Update user's inputs to point to last_post_node
            for j, inp in enumerate(user.inputs):
                if inp is source_node:
                    user.inputs[j] = last_post_node
            last_post_node.users.append(user)
        
        # Update ir_graph.outputs if source_node was an output
        for i, out_node in enumerate(ir_graph.outputs):
            if out_node is source_node:
                ir_graph.outputs[i] = last_post_node
    
    def _validate_float_output(self, ir_graph: IRGraph):
        """
        Validate that the final output is float32 for C API compatibility.
        
        The C API expects model_forward(float* input, float* output).
        If the output is not float32, the user must add DequantizeNode
        via their QuantIRNode.get_post_nodes() implementation.
        
        Raises:
            ValueError: If any output node is not float32
        """
        if not ir_graph.outputs:
            return
        
        for output_node in ir_graph.outputs:
            if output_node.dtype != 'float32':
                raise ValueError(
                    f"Graph output '{output_node.name}' has dtype '{output_node.dtype}', "
                    f"but C API requires float32 output. "
                    f"Ensure your QuantIRNode.get_post_nodes() returns a DequantizeNode."
                )
    
    def _get_expected_input_dtype(self, node: IRNode) -> str:
        """
        Get the expected input dtype for a node.
        
        Args:
            node: The node to check
            
        Returns:
            Expected input dtype ('float32', 'int8', or 'int16')
        """
        # Check if node has a dtype attribute indicating quantized
        if hasattr(node, 'dtype') and node.dtype in ['int8', 'int16']:
            return node.dtype
        
        # Default: float32
        return 'float32'
    
    def _quantize_weights(self, ir_graph: IRGraph, nodes_to_quantize: Dict[IRNode, QuantRule]):
        """
        Quantize weights during compilation.
        
        For each quantized node:
        1. Get float weights from ir_graph.parameters
        2. Use rule.quantize_weights() to quantize
        3. Replace float weights with quantized weights
        4. Store only quantized weights (design decision: save space)
        
        Args:
            ir_graph: The IR graph
            nodes_to_quantize: Dictionary mapping original node -> rule
        """
        # We need to process the NEW quantized nodes, not the old ones
        for node in ir_graph.nodes:
            if hasattr(node, 'scale') and hasattr(node, 'offset'):
                # This is a quantized node
                weight_name = node.metadata.get('weight_name')
                if not weight_name or weight_name not in ir_graph.parameters:
                    continue
                
                # Get the rule that applies to this node
                rule = None
                for orig_node, r in nodes_to_quantize.items():
                    if orig_node.name == node.name:
                        rule = r
                        break
                
                if rule is None:
                    continue
                
                # Get float weights
                weights_float = ir_graph.parameters[weight_name]
                
                # Quantize using rule's logic
                weights_q = rule.quantize_weights(weights_float)
                
                # Replace in parameters (only store quantized, not float)
                # Use same name (simpler than changing metadata)
                ir_graph.parameters[weight_name] = weights_q
    
    def _validate_graph(self, ir_graph: IRGraph):
        """
        Validate dtype compatibility in the graph.
        
        Ensures all nodes have compatible input dtypes.
        
        Args:
            ir_graph: The IR graph
            
        Raises:
            TypeError: If dtype validation fails
        """
        for node in ir_graph.nodes:
            try:
                if hasattr(node, 'validate_input_dtypes'):
                    node.validate_input_dtypes()
            except TypeError as e:
                raise TypeError(f"Dtype validation failed for node '{node.name}': {e}")

