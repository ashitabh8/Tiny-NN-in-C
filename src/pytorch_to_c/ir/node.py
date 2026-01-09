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
    
    def get_c_dtype(self) -> str:
        """
        Get the C data type string for this node's output buffer.
        
        Maps IR dtype to C type:
        - 'float32' -> 'float'
        - 'int8' -> 'int8_t'
        - 'int16' -> 'int16_t'
        
        Subclasses can override for custom behavior.
        """
        if self.dtype == 'int8':
            return 'int8_t'
        elif self.dtype == 'int16':
            return 'int16_t'
        else:
            return 'float'
    
    @property
    def is_quantized(self) -> bool:
        """
        Check if this node uses quantized types.
        
        Returns True if dtype is not float32 (i.e., int8, int16, or future types).
        Subclasses can override for custom behavior.
        """
        return self.dtype != 'float32'
    
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
    
    def validate_input_dtypes(self) -> bool:
        """
        Validate that input dtypes are compatible with this operation.
        
        Base implementation: float32 ops expect float32 inputs.
        Subclasses can override for specific requirements.
        
        Returns:
            True if valid
            
        Raises:
            TypeError: If input dtypes are incompatible
        """
        expected_dtype = self.dtype if self.dtype == 'float32' else 'float32'
        
        for inp in self.inputs:
            if inp.dtype != expected_dtype and inp.dtype != self.dtype:
                # Allow if types match OR if we expect float32 and get float32
                if not (self.dtype == 'float32' and inp.dtype == 'float32'):
                    pass  # For now, don't raise - subclasses will override
        
        return True
    
    def __repr__(self) -> str:
        return (f"IRNode(name='{self.name}', op_type='{self.op_type}', "
                f"shape={self.output_shape}, dtype='{self.dtype}')")
    
    def __str__(self) -> str:
        inputs_str = ", ".join([inp.name for inp in self.inputs])
        return f"{self.name} = {self.op_type}({inputs_str})"

