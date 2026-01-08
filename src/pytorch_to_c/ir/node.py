"""
IRNode - Intermediate Representation Node with double-linking
"""

from typing import List, Dict, Any, Optional, Tuple


class IRNode:
    """
    Represents a single operation in the IR graph.
    
    Double-linked: maintains both inputs and users for bidirectional traversal.
    This enables liveness analysis in Phase 3.
    """
    
    def __init__(
        self,
        name: str,
        op_type: str,
        output_shape: Optional[Tuple[int, ...]] = None,
        dtype: str = "float32",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an IR node.
        
        Args:
            name: Unique name for this node (from PyTorch layer name)
            op_type: Type of operation (e.g., 'conv2d', 'linear', 'relu')
            output_shape: Shape of the output tensor (if known)
            dtype: Data type ('float32', 'int8', 'int16')
            metadata: Additional operation-specific data (weights, attributes, etc.)
        """
        self.name = name
        self.op_type = op_type
        self.output_shape = output_shape
        self.dtype = dtype
        self.metadata = metadata or {}
        
        # Double-linking: both inputs and users
        self.inputs: List[IRNode] = []  # Nodes that this node depends on
        self.users: List[IRNode] = []   # Nodes that depend on this node
    
    def add_input(self, input_node: 'IRNode') -> None:
        """
        Add an input dependency to this node.
        Automatically updates the input node's users list.
        
        Args:
            input_node: The node that this node depends on
        """
        if input_node not in self.inputs:
            self.inputs.append(input_node)
            if self not in input_node.users:
                input_node.users.append(self)
    
    def add_user(self, user_node: 'IRNode') -> None:
        """
        Add a user (consumer) of this node's output.
        Automatically updates the user node's inputs list.
        
        Args:
            user_node: The node that depends on this node
        """
        if user_node not in self.users:
            self.users.append(user_node)
            if self not in user_node.inputs:
                user_node.inputs.append(self)
    
    def remove_input(self, input_node: 'IRNode') -> None:
        """Remove an input dependency."""
        if input_node in self.inputs:
            self.inputs.remove(input_node)
            if self in input_node.users:
                input_node.users.remove(self)
    
    def remove_user(self, user_node: 'IRNode') -> None:
        """Remove a user dependency."""
        if user_node in self.users:
            self.users.remove(user_node)
            if self in user_node.inputs:
                user_node.inputs.remove(self)
    
    def __repr__(self) -> str:
        return (f"IRNode(name='{self.name}', op_type='{self.op_type}', "
                f"shape={self.output_shape}, dtype='{self.dtype}')")
    
    def __str__(self) -> str:
        inputs_str = ", ".join([inp.name for inp in self.inputs])
        return f"{self.name} = {self.op_type}({inputs_str})"

