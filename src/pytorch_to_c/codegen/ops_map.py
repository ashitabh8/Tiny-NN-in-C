"""
Mapping from IR operations to C function implementations
"""

from typing import Dict, Any, List, Tuple


class OpMapping:
    """Maps IR operation types to C function calls."""
    
    # Map IR op_type to C function name
    OP_TO_C_FUNC = {
        'conv2d': 'conv2d_nhwc',
        'linear': 'dense',
        'relu': 'relu',
        'batchnorm': 'batchnorm2d_nhwc',
        'softmax': 'softmax',
        'add': 'add_tensors',
        'mul': 'mul_tensors',
    }
    
    @staticmethod
    def get_c_function(op_type: str) -> str:
        """
        Get the C function name for an IR operation type.
        
        Args:
            op_type: The IR operation type
            
        Returns:
            The corresponding C function name
        """
        if op_type not in OpMapping.OP_TO_C_FUNC:
            raise ValueError(f"Unsupported operation type: {op_type}")
        return OpMapping.OP_TO_C_FUNC[op_type]
    
    @staticmethod
    def get_function_signature(op_type: str) -> str:
        """
        Get a descriptive signature for documentation.
        
        Args:
            op_type: The IR operation type
            
        Returns:
            A string describing the function signature
        """
        signatures = {
            'conv2d': 'conv2d_nhwc(in, h, w, c_in, filt, k_h, k_w, c_out, bias, stride_h, stride_w, pad_same, out)',
            'linear': 'dense(x, in_features, W, b, out_features, y)',
            'relu': 'relu(x, n)',
            'batchnorm': 'batchnorm2d_nhwc(in, h, w, c, gamma, beta, mean, var, eps, out)',
            'softmax': 'softmax(x, n)',
            'add': 'add_tensors(a, b, n, out)',
            'mul': 'mul_tensors(a, b, n, out)',
        }
        return signatures.get(op_type, 'unknown')

