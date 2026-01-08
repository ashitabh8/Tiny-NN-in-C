"""
IRGraph - Container for the intermediate representation
"""

from typing import List, Dict, Optional, Any
import numpy as np

from .node import IRNode


class IRGraph:
    """
    Represents the complete computational graph in IR form.
    
    Maintains:
    - Topologically sorted list of nodes
    - Input and output nodes
    - Parameter storage (weights, biases, etc.)
    """
    
    def __init__(self):
        """Initialize an empty IR graph."""
        self.nodes: List[IRNode] = []
        self.inputs: List[IRNode] = []   # Input placeholder nodes
        self.outputs: List[IRNode] = []  # Output nodes
        self.parameters: Dict[str, np.ndarray] = {}  # name -> tensor data
        self._node_map: Dict[str, IRNode] = {}  # name -> node lookup
    
    def add_node(self, node: IRNode) -> None:
        """
        Add a node to the graph.
        
        Args:
            node: The IRNode to add
        """
        if node.name in self._node_map:
            raise ValueError(f"Node with name '{node.name}' already exists in graph")
        
        self.nodes.append(node)
        self._node_map[node.name] = node
    
    def get_node_by_name(self, name: str) -> Optional[IRNode]:
        """
        Retrieve a node by its name.
        
        Args:
            name: The name of the node to retrieve
            
        Returns:
            The IRNode if found, None otherwise
        """
        return self._node_map.get(name)
    
    def add_parameter(self, name: str, data: np.ndarray) -> None:
        """
        Add a parameter (weight, bias, etc.) to the graph.
        
        Args:
            name: Unique name for the parameter
            data: The parameter data as a numpy array
        """
        if name in self.parameters:
            raise ValueError(f"Parameter '{name}' already exists")
        self.parameters[name] = data
    
    def get_parameter(self, name: str) -> Optional[np.ndarray]:
        """
        Retrieve a parameter by name.
        
        Args:
            name: The parameter name
            
        Returns:
            The parameter data if found, None otherwise
        """
        return self.parameters.get(name)
    
    def mark_input(self, node: IRNode) -> None:
        """Mark a node as a graph input."""
        if node not in self.inputs:
            self.inputs.append(node)
    
    def mark_output(self, node: IRNode) -> None:
        """Mark a node as a graph output."""
        if node not in self.outputs:
            self.outputs.append(node)
    
    def topological_sort(self) -> List[IRNode]:
        """
        Perform topological sort on the graph.
        
        Note: In Phase 1, torch.fx already provides topological order,
        so this is primarily for validation or future use.
        
        Returns:
            List of nodes in topologically sorted order
        """
        visited = set()
        temp_mark = set()
        sorted_nodes = []
        
        def visit(node: IRNode):
            if node in temp_mark:
                raise ValueError(f"Graph contains a cycle at node '{node.name}'")
            if node in visited:
                return
            
            temp_mark.add(node)
            for input_node in node.inputs:
                visit(input_node)
            temp_mark.remove(node)
            visited.add(node)
            sorted_nodes.append(node)
        
        # Visit all nodes
        for node in self.nodes:
            if node not in visited:
                visit(node)
        
        return sorted_nodes
    
    def validate(self) -> bool:
        """
        Validate the graph structure.
        
        Returns:
            True if valid, raises exception otherwise
        """
        # Check for cycles
        try:
            self.topological_sort()
        except ValueError as e:
            raise ValueError(f"Graph validation failed: {e}")
        
        # Check that all input nodes exist
        for node in self.nodes:
            for input_node in node.inputs:
                if input_node not in self.nodes:
                    raise ValueError(
                        f"Node '{node.name}' has input '{input_node.name}' "
                        f"which is not in the graph"
                    )
        
        return True
    
    def __repr__(self) -> str:
        return (f"IRGraph(nodes={len(self.nodes)}, "
                f"inputs={len(self.inputs)}, "
                f"outputs={len(self.outputs)}, "
                f"parameters={len(self.parameters)})")
    
    def print_graph(self) -> str:
        """
        Generate a human-readable representation of the graph.
        
        Returns:
            String representation of the graph structure
        """
        lines = ["IRGraph:"]
        lines.append(f"  Inputs: {[n.name for n in self.inputs]}")
        lines.append(f"  Outputs: {[n.name for n in self.outputs]}")
        lines.append(f"  Parameters: {list(self.parameters.keys())}")
        lines.append("  Nodes:")
        
        for node in self.nodes:
            inputs_str = ", ".join([inp.name for inp in node.inputs])
            users_str = ", ".join([usr.name for usr in node.users])
            lines.append(f"    {node.name} [{node.op_type}]")
            lines.append(f"      inputs: [{inputs_str}]")
            lines.append(f"      users: [{users_str}]")
            lines.append(f"      shape: {node.output_shape}, dtype: {node.dtype}")
        
        return "\n".join(lines)

