"""General affine-linear quantized node (Phase 2 — delegates to existing kernels)."""

from __future__ import annotations

from typing import List, Optional

from ...ir.node import IRNode
from ...ir.quant_node import QuantIRNode
from ..quant_config import QuantLinearConfig


class QuantLinearNode(QuantIRNode):
    """
    Config-driven affine linear quant node.

    Phase 2: pre/post from config; codegen delegates to today's kernel call
    strings (byte-identical). Phase 3/4 replace the dispatch with templates.
    """

    def __init__(
        self,
        original_node: IRNode,
        config: QuantLinearConfig,
        *,
        input_scale: float = 1.0,
        output_scale: float = 1.0,
        weight_scale: float = 1.0,
        input_offset: int = 0,
        weight_offset: int = 0,
        output_offset: int = 0,
        resolved_group_size: Optional[int] = None,
    ):
        a_dtype = f"int{config.a_bits}"
        strategy = "dynamic" if config.dynamic_act else "static"
        if config.w_bits == 4:
            strategy = f"{strategy}_int4_per_group"

        super().__init__(
            original_node=original_node,
            dtype=a_dtype if not config.dynamic_act else "float32",
            scale=weight_scale,
            offset=weight_offset,
            quant_strategy=strategy,
        )
        self.config = config
        self.input_scale = input_scale
        self.output_scale = output_scale
        self.weight_scale = weight_scale
        self.input_offset = input_offset
        self.weight_offset = weight_offset
        self.output_offset = output_offset
        self.computation_dtype = a_dtype

        if config.dynamic_act:
            self.dtype = "float32"

        g = resolved_group_size or config.input_group_size
        self.resolved_group_size = g
        self.metadata["group_size"] = g
        self.metadata["quant_config"] = {
            "w_bits": config.w_bits,
            "a_bits": config.a_bits,
            "input_group_size": config.input_group_size,
            "per_out_column": config.per_out_column,
            "a_gran": config.a_gran.value,
            "w_symmetric": config.w_symmetric,
            "a_symmetric": config.a_symmetric,
            "dynamic_act": config.dynamic_act,
            "rounding": config.rounding,
        }

    def get_pre_nodes(self) -> List[IRNode]:
        if self.config.dynamic_act:
            from .quant_utils import DynamicQuantizeInputNode

            dtype = f"int{self.config.a_bits}"
            return [
                DynamicQuantizeInputNode(
                    name=f"{self.name}_input_dynq",
                    target_dtype=dtype,
                    output_shape=self.metadata.get("input_shape"),
                )
            ]

        if self.config.a_bits not in (8, 16):
            raise NotImplementedError(
                f"a_bits={self.config.a_bits} pre-node not implemented"
            )

        from .quant_utils import QuantizeNode

        dtype = f"int{self.config.a_bits}"
        return [
            QuantizeNode(
                name=f"{self.name}_input_q",
                target_dtype=dtype,
                scale=self.input_scale,
                offset=self.input_offset,
                output_shape=self.metadata.get("input_shape"),
            )
        ]

    def get_post_nodes(self) -> List[IRNode]:
        if self.config.dynamic_act:
            return []

        if self.config.a_bits not in (8, 16):
            raise NotImplementedError(
                f"a_bits={self.config.a_bits} post-node not implemented"
            )

        from .quant_utils import DequantizeNode

        dtype = f"int{self.config.a_bits}"
        return [
            DequantizeNode(
                name=f"{self.name}_output_dq",
                source_dtype=dtype,
                scale=self.output_scale,
                offset=self.output_offset,
                output_shape=self.output_shape,
            )
        ]

    def get_c_dtype(self) -> str:
        if self.config.dynamic_act:
            return "float"
        return super().get_c_dtype()

    def validate_input_dtypes(self) -> bool:
        if self.config.w_bits == 4:
            for inp in self.inputs:
                if inp.dtype != "int8":
                    raise TypeError(
                        f"QuantLinearNode '{self.name}' (W4A8) expects int8 input, "
                        f"got '{inp.dtype}' from '{inp.name}'"
                    )
            return True
        if self.config.dynamic_act:
            for inp in self.inputs:
                if inp.dtype not in ["int8", "int16"]:
                    raise TypeError(
                        f"QuantLinearNode '{self.name}' expects quantized input, "
                        f"got '{inp.dtype}' from '{inp.name}'"
                    )
            return True
        return super().validate_input_dtypes()

    def _get_input_scale_variable(self, printer) -> str:
        if self.inputs:
            input_node = self.inputs[0]
            if input_node.op_type == "dynamic_quantize":
                return f"scale_{printer._sanitize_name(input_node.name)}"
        raise ValueError(
            f"QuantLinearNode '{self.name}': expected input from "
            f"DynamicQuantizeInputNode (op_type='dynamic_quantize'), but "
            f"got '{self.inputs[0].op_type if self.inputs else 'none'}' "
            f"from '{self.inputs[0].name if self.inputs else 'N/A'}'."
        )

    def _is_per_group(self) -> bool:
        if self.metadata.get("per_group_weight_scales_param"):
            return True
        if self.metadata.get("per_channel_weight_scales_param"):
            return False
        in_features = self.metadata.get("in_features", self.config.input_group_size)
        return (
            self.config.per_out_column
            and self.config.input_group_size < in_features
        )

    def generate_c_code(self, c_printer) -> List[str]:
        cfg = self.config
        input_buffer = c_printer._get_input_buffer(self, 0)
        output_buffer = c_printer._get_buffer_name(self)
        weight_name = c_printer._sanitize_name(self.metadata["weight_name"])
        bias_name = (
            c_printer._sanitize_name(self.metadata["bias_name"])
            if self.metadata.get("bias_name")
            else "NULL"
        )
        in_features = self.metadata["in_features"]
        out_features = self.metadata["out_features"]

        if cfg.w_bits == 4:
            return self._c_int4(c_printer, input_buffer, output_buffer,
                                weight_name, bias_name, in_features, out_features)

        bits = cfg.a_bits
        if cfg.dynamic_act:
            input_scale_var = self._get_input_scale_variable(c_printer)
            fn = f"dense_int{bits}_to_float"
            return [
                f"{fn}("
                f"{input_buffer}, {in_features}, "
                f"{weight_name}, {bias_name}, {out_features}, "
                f"{input_scale_var}, {self.weight_scale}f, "
                f"{output_buffer});"
            ]

        if not cfg.per_out_column:
            fn = f"dense_int{bits}"
            return [
                f"{fn}("
                f"{input_buffer}, {in_features}, "
                f"{weight_name}, {bias_name}, {out_features}, "
                f"{self.input_scale}f, {self.weight_scale}f, {self.output_scale}f, "
                f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, "
                f"{output_buffer});"
            ]

        if self._is_per_group():
            scales_param = self.metadata.get("per_group_weight_scales_param")
            group_size = self.metadata.get("group_size")
            if not scales_param or group_size is None:
                raise ValueError(
                    f"QuantLinearNode '{self.name}': missing group metadata"
                )
            scales_c = c_printer._sanitize_name(scales_param)
            fn = f"dense_int{bits}_per_group"
            return [
                f"{fn}("
                f"{input_buffer}, {in_features}, "
                f"{weight_name}, {bias_name}, {out_features}, "
                f"{group_size}, {self.input_scale}f, {scales_c}, {self.output_scale}f, "
                f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, "
                f"{output_buffer});"
            ]

        scales_param = self.metadata.get("per_channel_weight_scales_param")
        if not scales_param:
            raise ValueError(
                f"QuantLinearNode '{self.name}': missing "
                f"metadata['per_channel_weight_scales_param']"
            )
        scales_c = c_printer._sanitize_name(scales_param)
        fn = f"dense_int{bits}_per_channel"
        return [
            f"{fn}("
            f"{input_buffer}, {in_features}, "
            f"{weight_name}, {bias_name}, {out_features}, "
            f"{self.input_scale}f, {scales_c}, {self.output_scale}f, "
            f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, "
            f"{output_buffer});"
        ]

    def _c_int4(self, c_printer, input_buffer, output_buffer,
                weight_name, bias_name, in_features, out_features) -> List[str]:
        scales_param = self.metadata.get("per_group_weight_scales_param")
        group_size = self.metadata.get("group_size", self.resolved_group_size)
        count = self.metadata.get("packed_weight_count")
        if not scales_param or group_size is None or count is None:
            raise ValueError(
                f"QuantLinearNode '{self.name}': missing int4 group metadata"
            )
        scales_c = c_printer._sanitize_name(scales_param)
        if self.config.dynamic_act:
            input_scale_var = self._get_input_scale_variable(c_printer)
            return [
                f"dense_int8_int4w_per_group_to_float("
                f"{input_buffer}, {in_features}, "
                f"{weight_name}, {count}, {bias_name}, {out_features}, "
                f"{group_size}, {input_scale_var}, {scales_c}, "
                f"{output_buffer});"
            ]
        return [
            f"dense_int8_int4w_per_group("
            f"{input_buffer}, {in_features}, "
            f"{weight_name}, {count}, {bias_name}, {out_features}, "
            f"{group_size}, {self.input_scale}f, {scales_c}, {self.output_scale}f, "
            f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, "
            f"{output_buffer});"
        ]

    def generate_triton_code(self, printer) -> List[str]:
        cfg = self.config
        input_buffer = printer._get_input_buffer(self, 0)
        output_buffer = printer._get_buffer_name(self)
        weight_name = printer._w(self.metadata["weight_name"])
        bias_name = (
            printer._w(self.metadata["bias_name"])
            if self.metadata.get("bias_name")
            else "None"
        )
        in_features = self.metadata["in_features"]
        out_features = self.metadata["out_features"]

        if cfg.w_bits == 4:
            return self._triton_int4(
                printer, input_buffer, output_buffer,
                weight_name, bias_name, in_features, out_features,
            )

        bits = cfg.a_bits
        if cfg.dynamic_act:
            input_scale_var = self._get_input_scale_variable(printer)
            fn = f"dense_int{bits}_to_float"
            return [
                f"ops_q.{fn}({input_buffer}, {in_features}, {weight_name}, {bias_name}, "
                f"{out_features}, {input_scale_var}, {self.weight_scale}, {output_buffer})"
            ]

        if not cfg.per_out_column:
            fn = f"dense_int{bits}"
            return [
                f"ops_q.{fn}({input_buffer}, {in_features}, {weight_name}, {bias_name}, "
                f"{out_features}, {self.input_scale}, {self.weight_scale}, {self.output_scale}, "
                f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, {output_buffer})"
            ]

        if self._is_per_group():
            scales_param = self.metadata.get("per_group_weight_scales_param")
            group_size = self.metadata.get("group_size")
            scales_c = printer._w(scales_param)
            fn = f"dense_int{bits}_per_group"
            return [
                f"ops_q.{fn}({input_buffer}, {in_features}, {weight_name}, {bias_name}, "
                f"{out_features}, {group_size}, {self.input_scale}, {scales_c}, "
                f"{self.output_scale}, {self.input_offset}, {self.weight_offset}, "
                f"{self.output_offset}, {output_buffer})"
            ]

        scales_param = self.metadata.get("per_channel_weight_scales_param")
        if not scales_param:
            raise ValueError(
                f"QuantLinearNode '{self.name}': missing per_channel scales"
            )
        scales_c = printer._w(scales_param)
        fn = f"dense_int{bits}_per_channel"
        return [
            f"ops_q.{fn}({input_buffer}, {in_features}, {weight_name}, {bias_name}, "
            f"{out_features}, {self.input_scale}, {scales_c}, {self.output_scale}, "
            f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, {output_buffer})"
        ]

    def _triton_int4(self, printer, input_buffer, output_buffer,
                     weight_name, bias_name, in_features, out_features) -> List[str]:
        scales_param = self.metadata.get("per_group_weight_scales_param")
        group_size = self.metadata.get("group_size", self.resolved_group_size)
        count = self.metadata.get("packed_weight_count")
        scales_c = printer._w(scales_param)
        if self.config.dynamic_act:
            input_scale_var = self._get_input_scale_variable(printer)
            return [
                f"ops_q.dense_int8_int4w_per_group_to_float({input_buffer}, {in_features}, "
                f"{weight_name}, {count}, {bias_name}, {out_features}, "
                f"{group_size}, {input_scale_var}, {scales_c}, {output_buffer})"
            ]
        return [
            f"ops_q.dense_int8_int4w_per_group({input_buffer}, {in_features}, "
            f"{weight_name}, {count}, {bias_name}, {out_features}, "
            f"{group_size}, {self.input_scale}, {scales_c}, {self.output_scale}, "
            f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, "
            f"{output_buffer})"
        ]

    @property
    def matmul_output_dtype(self) -> str:
        if self.config.dynamic_act:
            return "float32"
        return f"int{self.config.a_bits}"
