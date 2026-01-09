"""
Unit tests for quantization module (Phase 2.1)
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pytorch_to_c.ir import IRNode, QuantIRNode
from src.pytorch_to_c.quantization import (
    QuantRule, StaticQuantRule, DynamicQuantRuleMinMaxPerTensor,
    RuleMatcher, QuantizeNode, DequantizeNode
)


class TestStaticQuantRule:
    """Test StaticQuantRule"""
    
    def test_rule_creation(self):
        """Test creating a static quantization rule"""
        rule = StaticQuantRule(
            r'.*fc.*', 'int8',
            input_scale=0.05, input_offset=0,
            weight_scale=0.05, weight_offset=0,
            output_scale=0.05, output_offset=0
        )
        
        assert rule.pattern == r'.*fc.*'
        assert rule.dtype == 'int8'
        assert rule.input_scale == 0.05
        assert rule.weight_scale == 0.05
        assert rule.output_scale == 0.05
    
    def test_rule_matches_pattern(self):
        """Test pattern matching on node names"""
        rule = StaticQuantRule(
            r'.*fc.*', 'int8',
            input_scale=0.05, input_offset=0,
            weight_scale=0.05, weight_offset=0,
            output_scale=0.05, output_offset=0
        )
        
        # Should match
        fc_node = IRNode(name='fc1', op_type='linear')
        assert rule.matches(fc_node) is True
        
        fc2_node = IRNode(name='model_fc_layer', op_type='linear')
        assert rule.matches(fc2_node) is True
        
        # Should not match
        conv_node = IRNode(name='conv1', op_type='conv2d')
        assert rule.matches(conv_node) is False
        
        relu_node = IRNode(name='relu', op_type='relu')
        assert rule.matches(relu_node) is False
    
    def test_quantize_weights_int8(self):
        """Test weight quantization to int8"""
        rule = StaticQuantRule(
            r'.*', 'int8',
            input_scale=0.1, input_offset=0,
            weight_scale=0.1, weight_offset=0,
            output_scale=0.1, output_offset=0
        )
        
        weights = np.array([0.0, 0.1, 0.2, -0.1, -0.2], dtype=np.float32)
        weights_q = rule.quantize_weights(weights)
        
        assert weights_q.dtype == np.int8
        assert len(weights_q) == len(weights)
        np.testing.assert_array_equal(weights_q, [0, 1, 2, -1, -2])
    
    def test_quantize_weights_int16(self):
        """Test weight quantization to int16"""
        rule = StaticQuantRule(
            r'.*', 'int16',
            input_scale=0.001, input_offset=0,
            weight_scale=0.001, weight_offset=0,
            output_scale=0.001, output_offset=0
        )
        
        weights = np.array([0.0, 1.0, 2.0, -1.0, -2.0], dtype=np.float32)
        weights_q = rule.quantize_weights(weights)
        
        assert weights_q.dtype == np.int16
        np.testing.assert_array_equal(weights_q, [0, 1000, 2000, -1000, -2000])
    
    def test_quantize_weights_clipping(self):
        """Test that weights are clipped to valid range"""
        rule = StaticQuantRule(
            r'.*', 'int8',
            input_scale=0.01, input_offset=0,
            weight_scale=0.01, weight_offset=0,
            output_scale=0.01, output_offset=0
        )
        
        # Values that would exceed int8 range
        weights = np.array([10.0, -10.0], dtype=np.float32)  # Would be 1000, -1000 without clipping
        weights_q = rule.quantize_weights(weights)
        
        assert weights_q.dtype == np.int8
        assert weights_q[0] == 127  # Clipped to max
        assert weights_q[1] == -128  # Clipped to min
    
    def test_get_quant_params(self):
        """Test getting quantization parameters"""
        rule = StaticQuantRule(
            r'.*fc.*', 'int8',
            input_scale=0.05, input_offset=5,
            weight_scale=0.05, weight_offset=5,
            output_scale=0.05, output_offset=5
        )
        params = rule.get_quant_params()
        
        assert params['dtype'] == 'int8'
        assert params['input_scale'] == 0.05
        assert params['weight_scale'] == 0.05
        assert params['output_scale'] == 0.05


class TestDynamicQuantRuleMinMaxPerTensor:
    """Test DynamicQuantRuleMinMaxPerTensor"""
    
    def test_rule_creation(self):
        """Test creating a dynamic quantization rule"""
        rule = DynamicQuantRuleMinMaxPerTensor(r'.*fc.*', 'int8')
        
        assert rule.pattern == r'.*fc.*'
        assert rule.dtype == 'int8'
    
    def test_quantize_weights_computes_scale(self):
        """Test that dynamic rule computes scale/offset from weights"""
        rule = DynamicQuantRuleMinMaxPerTensor(r'.*', 'int8')
        
        # Weights from -1 to 1
        weights = np.linspace(-1, 1, 100, dtype=np.float32)
        weights_q = rule.quantize_weights(weights)
        
        assert weights_q.dtype == np.int8
        assert weights_q.min() >= -128
        assert weights_q.max() <= 127


class TestRuleMatcher:
    """Test RuleMatcher"""
    
    def test_find_matching_rule(self):
        """Test finding matching rules"""
        rules = [
            StaticQuantRule(
                r'.*fc.*', 'int8',
                input_scale=0.05, input_offset=0,
                weight_scale=0.05, weight_offset=0,
                output_scale=0.05, output_offset=0
            ),
            StaticQuantRule(
                r'.*conv.*', 'int16',
                input_scale=0.01, input_offset=0,
                weight_scale=0.01, weight_offset=0,
                output_scale=0.01, output_offset=0
            ),
        ]
        matcher = RuleMatcher(rules)
        
        # FC should match first rule
        fc_node = IRNode(name='fc1', op_type='linear')
        rule = matcher.find_matching_rule(fc_node)
        assert rule is not None
        assert rule.dtype == 'int8'
        
        # Conv should match second rule
        conv_node = IRNode(name='conv1', op_type='conv2d')
        rule = matcher.find_matching_rule(conv_node)
        assert rule is not None
        assert rule.dtype == 'int16'
        
        # Relu should not match
        relu_node = IRNode(name='relu', op_type='relu')
        rule = matcher.find_matching_rule(relu_node)
        assert rule is None
    
    def test_first_match_wins(self):
        """Test that first matching rule is returned"""
        rules = [
            StaticQuantRule(
                r'.*', 'int8',
                input_scale=0.05, input_offset=0,
                weight_scale=0.05, weight_offset=0,
                output_scale=0.05, output_offset=0
            ),  # Matches everything
            StaticQuantRule(
                r'.*fc.*', 'int16',
                input_scale=0.01, input_offset=0,
                weight_scale=0.01, weight_offset=0,
                output_scale=0.01, output_offset=0
            ),  # More specific
        ]
        matcher = RuleMatcher(rules)
        
        # First rule should match even though second is more specific
        fc_node = IRNode(name='fc1', op_type='linear')
        rule = matcher.find_matching_rule(fc_node)
        assert rule.dtype == 'int8'


class TestQuantizeNode:
    """Test QuantizeNode"""
    
    def test_creation(self):
        """Test creating a QuantizeNode"""
        node = QuantizeNode(
            name='test_quantize',
            target_dtype='int8',
            scale=0.05,
            offset=0
        )
        
        assert node.name == 'test_quantize'
        assert node.op_type == 'quantize'
        assert node.dtype == 'int8'
        assert node.target_dtype == 'int8'
        assert node.scale == 0.05
        assert node.offset == 0
    
    def test_get_c_dtype(self):
        """Test getting C dtype"""
        node_int8 = QuantizeNode('q1', 'int8', 0.05, 0)
        assert node_int8.get_c_dtype() == 'int8_t'
        
        node_int16 = QuantizeNode('q2', 'int16', 0.01, 0)
        assert node_int16.get_c_dtype() == 'int16_t'
    
    def test_validate_float_input(self):
        """Test input dtype validation"""
        q_node = QuantizeNode('q1', 'int8', 0.05, 0)
        
        # Valid: float32 input
        float_node = IRNode(name='input', op_type='input', dtype='float32')
        q_node.inputs = [float_node]
        assert q_node.validate_input_dtypes() is True
        
        # Invalid: int8 input
        int_node = IRNode(name='input', op_type='input', dtype='int8')
        q_node.inputs = [int_node]
        with pytest.raises(TypeError):
            q_node.validate_input_dtypes()


class TestDequantizeNode:
    """Test DequantizeNode"""
    
    def test_creation(self):
        """Test creating a DequantizeNode"""
        node = DequantizeNode(
            name='test_dequantize',
            source_dtype='int8',
            scale=0.05,
            offset=0
        )
        
        assert node.name == 'test_dequantize'
        assert node.op_type == 'dequantize'
        assert node.dtype == 'float32'  # Output is always float32
        assert node.source_dtype == 'int8'
        assert node.scale == 0.05
        assert node.offset == 0
    
    def test_get_c_dtype(self):
        """Test getting C dtype"""
        node = DequantizeNode('dq1', 'int8', 0.05, 0)
        assert node.get_c_dtype() == 'float'
    
    def test_validate_quantized_input(self):
        """Test input dtype validation"""
        dq_node = DequantizeNode('dq1', 'int8', 0.05, 0)
        
        # Valid: int8 input
        int_node = IRNode(name='input', op_type='input', dtype='int8')
        dq_node.inputs = [int_node]
        assert dq_node.validate_input_dtypes() is True
        
        # Invalid: float32 input
        float_node = IRNode(name='input', op_type='input', dtype='float32')
        dq_node.inputs = [float_node]
        with pytest.raises(TypeError):
            dq_node.validate_input_dtypes()


class TestIRNodeDtype:
    """Test IRNode dtype field and validation"""
    
    def test_default_dtype(self):
        """Test default dtype is float32"""
        node = IRNode(name='test', op_type='linear')
        assert node.dtype == 'float32'
    
    def test_custom_dtype(self):
        """Test setting custom dtype"""
        node = IRNode(name='test', op_type='linear', dtype='int8')
        assert node.dtype == 'int8'
    
    def test_dtype_in_repr(self):
        """Test dtype appears in repr"""
        node = IRNode(name='test', op_type='linear', dtype='int8')
        assert 'int8' in repr(node)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

