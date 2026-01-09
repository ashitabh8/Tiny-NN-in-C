"""
Base class for IR optimization passes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from src.pytorch_to_c.ir.graph import IRGraph


class IRPass(ABC):
    """
    Abstract base class for IR optimization passes.
    
    All passes should inherit from this class and implement the apply() method.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the pass.
        
        Args:
            verbose: If True, print information about optimizations applied
        """
        self.verbose = verbose
        self.stats: Dict[str, Any] = {}
    
    @abstractmethod
    def apply(self, ir_graph: IRGraph) -> IRGraph:
        """
        Apply the optimization pass to the IR graph.
        
        Args:
            ir_graph: The IR graph to optimize
            
        Returns:
            The optimized IR graph (may be modified in-place)
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the last pass application.
        
        Returns:
            Dictionary with pass statistics
        """
        return self.stats
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")

