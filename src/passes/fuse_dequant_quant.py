"""
Pass to fuse/eliminate consecutive Dequantize → Quantize pairs.

When a DequantizeNode (intX → float32) is immediately followed by a QuantizeNode
(float32 → intX) of the SAME dtype AND SAME scale/offset, this is a true no-op
and can be eliminated.

Example before optimization:
    encoder_fc1 (int8) → dequantize (float32) → quantize (int8) → encoder_fc2 (int8)

After optimization:
    encoder_fc1 (int8) → encoder_fc2 (int8)

This saves:
- 2 function calls (dequantize + quantize)
- 1 intermediate float32 buffer
- Computation time

Note: For static quantization with matching scales, this is numerically exact.
      For dynamic quantization, scales may differ and this pass should not fuse.
"""

from typing import List, Tuple
import math

from .base import IRPass
from src.pytorch_to_c.ir.graph import IRGraph
from src.pytorch_to_c.ir.node import IRNode
from src.pytorch_to_c.quantization.ops.quant_utils import (
    QuantizeNode, 
    DequantizeNode,
    DynamicQuantizeInputNode
)


class FuseDequantQuantPass(IRPass):
    """
    Eliminates consecutive Dequantize → Quantize pairs of the same dtype.
    
    This pass identifies patterns where:
    1. A DequantizeNode converts from intX to float32
    2. The ONLY user of that DequantizeNode is a QuantizeNode (or DynamicQuantizeInputNode)
    3. The QuantizeNode converts back to the SAME intX type
    
    When found, both nodes are removed and the graph is rewired to bypass them.
    """
    
    def apply(self, ir_graph: IRGraph) -> IRGraph:
        """
        Apply the dequant-quant fusion pass.
        
        Args:
            ir_graph: The IR graph to optimize
            
        Returns:
            The optimized IR graph
        """
        self.stats = {
            'pairs_found': 0,
            'pairs_fused': 0,
            'nodes_removed': 0,
            'fused_pairs': []
        }
        
        # Find all fusable pairs
        pairs_to_fuse = self._find_fusable_pairs(ir_graph)
        
        self.stats['pairs_found'] = len(pairs_to_fuse)
        
        # Fuse each pair (process in reverse order to maintain indices)
        for dequant_node, quant_node in reversed(pairs_to_fuse):
            self._fuse_pair(ir_graph, dequant_node, quant_node)
            self.stats['pairs_fused'] += 1
            self.stats['nodes_removed'] += 2
            self.stats['fused_pairs'].append((dequant_node.name, quant_node.name))
            
            self._log(f"Fused: {dequant_node.name} → {quant_node.name}")
        
        if self.stats['pairs_fused'] > 0:
            self._log(f"Total: Removed {self.stats['nodes_removed']} nodes")
        
        return ir_graph
    
    def _find_fusable_pairs(self, ir_graph: IRGraph) -> List[Tuple[DequantizeNode, IRNode]]:
        """
        Find all Dequantize → Quantize pairs that can be fused.
        
        Criteria:
        1. DequantizeNode followed by QuantizeNode (NOT DynamicQuantizeInputNode)
        2. DequantizeNode has exactly ONE user (the quantize node)
        3. Both convert to/from the SAME quantized dtype (int8 or int16)
        4. Scale and offset must match (for numerical correctness)
        
        Returns:
            List of (dequant_node, quant_node) tuples to fuse
        """
        pairs = []
        
        for node in ir_graph.nodes:
            # Look for DequantizeNode
            if not isinstance(node, DequantizeNode):
                continue
            
            dequant_node = node
            
            # Check: exactly one user
            if len(dequant_node.users) != 1:
                self._log(f"Skipping {dequant_node.name}: has {len(dequant_node.users)} users")
                continue
            
            # Check: user is a static QuantizeNode (not dynamic)
            user = dequant_node.users[0]
            if isinstance(user, DynamicQuantizeInputNode):
                self._log(f"Skipping {dequant_node.name} → {user.name}: "
                         f"dynamic quantization cannot be fused (scale computed at runtime)")
                continue
            
            if not isinstance(user, QuantizeNode):
                continue
            
            quant_node = user
            
            # Check: same quantized dtype
            dequant_source_dtype = self._get_dequant_source_dtype(dequant_node)
            quant_target_dtype = quant_node.dtype
            
            if dequant_source_dtype != quant_target_dtype:
                self._log(f"Skipping {dequant_node.name} → {quant_node.name}: "
                         f"dtype mismatch ({dequant_source_dtype} → {quant_target_dtype})")
                continue
            
            # Check: scales and offsets must match for true no-op
            if not self._scales_match(dequant_node, quant_node):
                self._log(f"Skipping {dequant_node.name} → {quant_node.name}: "
                         f"scale/offset mismatch (dequant: scale={dequant_node.scale}, offset={dequant_node.offset}, "
                         f"quant: scale={quant_node.scale}, offset={quant_node.offset})")
                continue
            
            # This pair can be fused!
            pairs.append((dequant_node, quant_node))
            self._log(f"Found fusable pair: {dequant_node.name} → {quant_node.name} "
                     f"(dtype={dequant_source_dtype}, scale={dequant_node.scale}, offset={dequant_node.offset})")
        
        return pairs
    
    def _scales_match(self, dequant_node: DequantizeNode, quant_node: QuantizeNode, 
                      rel_tol: float = 1e-6) -> bool:
        """
        Check if scale and offset match between dequantize and quantize nodes.
        
        Args:
            dequant_node: The dequantize node
            quant_node: The quantize node
            rel_tol: Relative tolerance for floating point comparison
            
        Returns:
            True if scales and offsets match
        """
        # Compare scales with relative tolerance
        scale_match = math.isclose(dequant_node.scale, quant_node.scale, rel_tol=rel_tol)
        
        # Compare offsets (integer comparison)
        offset_match = dequant_node.offset == quant_node.offset
        
        return scale_match and offset_match
    
    def _get_dequant_source_dtype(self, dequant_node: DequantizeNode) -> str:
        """Get the source (quantized) dtype of a DequantizeNode."""
        # The input to dequantize is the quantized tensor
        if dequant_node.inputs:
            return dequant_node.inputs[0].dtype
        return 'unknown'
    
    def _fuse_pair(self, ir_graph: IRGraph, dequant_node: DequantizeNode, quant_node: IRNode):
        """
        Remove a Dequantize → Quantize pair and rewire the graph.
        
        Before: A → dequant → quant → B
        After:  A → B
        
        Args:
            ir_graph: The IR graph
            dequant_node: The DequantizeNode to remove
            quant_node: The QuantizeNode to remove
        """
        # Get the node BEFORE dequant (the actual quantized data source)
        source_node = dequant_node.inputs[0] if dequant_node.inputs else None
        
        # Get the nodes AFTER quant (the consumers of the quantized data)
        target_nodes = quant_node.users.copy()
        
        if source_node is None:
            self._log(f"Warning: {dequant_node.name} has no input, skipping")
            return
        
        # Rewire: source_node → target_nodes (bypass both dequant and quant)
        
        # 1. Remove dequant from source_node's users
        if dequant_node in source_node.users:
            source_node.users.remove(dequant_node)
        
        # 2. Connect source_node directly to target_nodes
        for target in target_nodes:
            # Update target's inputs: replace quant_node with source_node
            for i, inp in enumerate(target.inputs):
                if inp is quant_node:
                    target.inputs[i] = source_node
            
            # Add target to source_node's users
            if target not in source_node.users:
                source_node.users.append(target)
        
        # 3. Remove both nodes from graph
        if dequant_node in ir_graph.nodes:
            ir_graph.nodes.remove(dequant_node)
        if quant_node in ir_graph.nodes:
            ir_graph.nodes.remove(quant_node)
        
        # 4. Update graph outputs if needed
        if dequant_node in ir_graph.outputs:
            idx = ir_graph.outputs.index(dequant_node)
            ir_graph.outputs[idx] = source_node
        if quant_node in ir_graph.outputs:
            idx = ir_graph.outputs.index(quant_node)
            ir_graph.outputs[idx] = source_node

