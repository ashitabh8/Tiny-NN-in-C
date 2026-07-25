"""
LQER - Low-rank Quantization Error Reconstruction as IR nodes.

LQER corrects the output of a quantized layer with a low-rank approximation
of the weight quantization error. The user supplies the FULL error matrix
E (their responsibility to compute, e.g. E = W - dequant(quant(W))) and a
rank r; the node factorizes E ~= A @ B at compile time via a pluggable
factorizer (default: truncated SVD; swap in PCA or anything else with the
same signature).

Runtime dataflow (dynamic-quant linear example):

    x (float32) --+--> DynamicQuantizeInput --> DynamicQuantLinear --+
                  |                                                  v
                  +--> LQERMatmul(x @ A) --> LQERMatmul(@ B) --> LQERAdd --> users

Everything is an explicit IR node (nothing hidden in generated code):
- The error branch taps the ORIGINAL float input, so it is topologically
  independent of the quantized path; the C backend runs the two chains in
  parallel via OpenMP sections (see codegen/parallel_regions.py).
- The branch matmuls reuse the existing float `dense`/`conv2d_nhwc` codegen
  (op_type 'linear' / 'conv2d'), so no new kernels are needed.

For conv2d, conv(x, E) == (im2col(x) @ A) @ B, so the branch is a float
conv with r output channels (kernel = A reshaped [kh, kw, in_c, r]) followed
by a 1x1 float conv (kernel = B as [1, 1, r, out_c]).

Changing the correction scheme means editing THIS file only - the graph
transform contains no LQER knowledge (it just wires any declared diamond).
"""

from typing import Callable, List, Optional, Tuple

import numpy as np

from ...ir.node import IRNode
from .quant_linear import DynamicQuantLinearNode, StaticQuantLinearNode
from .quant_conv2d import DynamicQuantConv2dNode, StaticQuantConv2dNode


# ---------------------------------------------------------------------------
# Factorizers
# ---------------------------------------------------------------------------

def svd_factorizer(error_matrix_2d: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Default factorizer: truncated SVD.

    E [K, N] ~= A @ B with A = U_r * s_r  [K, r], B = V_r^T  [r, N].

    Any callable with this signature can replace it (PCA, randomized SVD, ...).
    """
    U, s, Vt = np.linalg.svd(np.asarray(error_matrix_2d, dtype=np.float64),
                             full_matrices=False)
    r = max(1, min(int(rank), len(s)))
    A = (U[:, :r] * s[:r]).astype(np.float32)
    B = Vt[:r, :].astype(np.float32)
    return A, B


# ---------------------------------------------------------------------------
# Branch / join nodes
# ---------------------------------------------------------------------------

class LQERMatmulNode(IRNode):
    """
    Float matmul of the LQER error branch (x @ A or intermediate @ B).

    op_type is 'linear' so the existing float dense codegen (C and Triton)
    emits it unchanged; the CLASS is what identifies it to passes
    (isinstance) and carries LQER-specific state (role, rank, corrected
    layer). Tagged with metadata['parallel_branch'] for region detection.
    """

    def __init__(
        self,
        name: str,
        in_features: int,
        out_features: int,
        weight_name: str,
        role: str,
        corrected_layer: str,
        rank: int,
        output_shape: Optional[Tuple[int, ...]] = None,
    ):
        if role not in ('down', 'up'):
            raise ValueError(f"LQERMatmulNode role must be 'down' or 'up', got '{role}'")
        super().__init__(
            name=name,
            op_type='linear',
            output_shape=output_shape,
            dtype='float32',
            metadata={
                'weight_name': weight_name,
                'bias_name': None,
                'in_features': in_features,
                'out_features': out_features,
                'parallel_branch': True,
                'lqer_role': role,
                'lqer_corrected_layer': corrected_layer,
                'lqer_rank': rank,
            }
        )
        self.role = role
        self.rank = rank
        self.corrected_layer = corrected_layer

    def validate_input_dtypes(self) -> bool:
        for inp in self.inputs:
            if inp.dtype != 'float32':
                raise TypeError(
                    f"LQERMatmulNode '{self.name}' expects float32 input "
                    f"(the branch taps the pre-quantization activation), "
                    f"got '{inp.dtype}' from '{inp.name}'"
                )
        return True

    def __repr__(self) -> str:
        return (f"LQERMatmulNode(name='{self.name}', role='{self.role}', "
                f"in={self.metadata['in_features']}, out={self.metadata['out_features']}, "
                f"rank={self.rank}, corrects='{self.corrected_layer}')")


class LQERConvMatmulNode(IRNode):
    """
    Float conv of the LQER error branch for conv2d layers.

    'down' role: conv with r output channels, kernel = A reshaped
    [kh, kw, in_c, r], stride/padding copied from the corrected conv.
    'up' role: 1x1 conv, kernel = B as [1, 1, r, out_c].

    op_type is 'conv2d' so the existing float conv2d_nhwc codegen emits it.
    """

    def __init__(
        self,
        name: str,
        kernel_size: Tuple[int, int],
        stride,
        padding,
        in_channels: int,
        out_channels: int,
        weight_name: str,
        role: str,
        corrected_layer: str,
        rank: int,
        output_shape: Optional[Tuple[int, ...]] = None,
    ):
        if role not in ('down', 'up'):
            raise ValueError(f"LQERConvMatmulNode role must be 'down' or 'up', got '{role}'")
        super().__init__(
            name=name,
            op_type='conv2d',
            output_shape=output_shape,
            dtype='float32',
            metadata={
                'weight_name': weight_name,
                'bias_name': None,
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'in_channels': in_channels,
                'out_channels': out_channels,
                'groups': 1,
                'parallel_branch': True,
                'lqer_role': role,
                'lqer_corrected_layer': corrected_layer,
                'lqer_rank': rank,
            }
        )
        self.role = role
        self.rank = rank
        self.corrected_layer = corrected_layer

    def validate_input_dtypes(self) -> bool:
        for inp in self.inputs:
            if inp.dtype != 'float32':
                raise TypeError(
                    f"LQERConvMatmulNode '{self.name}' expects float32 input, "
                    f"got '{inp.dtype}' from '{inp.name}'"
                )
        return True

    def __repr__(self) -> str:
        return (f"LQERConvMatmulNode(name='{self.name}', role='{self.role}', "
                f"in_ch={self.metadata['in_channels']}, out_ch={self.metadata['out_channels']}, "
                f"rank={self.rank}, corrects='{self.corrected_layer}')")


class LQERAddNode(IRNode):
    """
    Join node: elementwise float add of the quantized path output and the
    low-rank error correction. Exactly two inputs:
    inputs[0] = quantized path (float32 after dequant / float-output kernel),
    inputs[1] = error branch output.

    op_type is 'add' so the existing elementwise-add codegen emits it.
    metadata['parallel_join'] closes the parallel region for the backends.
    """

    def __init__(
        self,
        name: str,
        corrected_layer: str,
        output_shape: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__(
            name=name,
            op_type='add',
            output_shape=output_shape,
            dtype='float32',
            metadata={
                'parallel_join': True,
                'lqer_corrected_layer': corrected_layer,
            }
        )
        self.corrected_layer = corrected_layer

    def validate_input_dtypes(self) -> bool:
        if len(self.inputs) != 2:
            raise TypeError(
                f"LQERAddNode '{self.name}' expects exactly 2 inputs "
                f"(quant path, error branch), got {len(self.inputs)}"
            )
        for inp in self.inputs:
            if inp.dtype != 'float32':
                raise TypeError(
                    f"LQERAddNode '{self.name}' expects float32 inputs, "
                    f"got '{inp.dtype}' from '{inp.name}'"
                )
        return True

    def __repr__(self) -> str:
        return f"LQERAddNode(name='{self.name}', corrects='{self.corrected_layer}')"


# ---------------------------------------------------------------------------
# Branch declaration mixins
# ---------------------------------------------------------------------------

class _LQERBranchBase:
    """Shared LQER state: error matrix, rank, factorizer, param registration."""

    def _init_lqer(
        self,
        error_matrix: np.ndarray,
        rank: int,
        factorizer: Optional[Callable] = None,
    ) -> None:
        if error_matrix is None:
            raise ValueError(
                f"LQER node '{self.name}': error_matrix is required "
                f"(compute it as W_float - dequant(quant(W_float)))"
            )
        if int(rank) < 1:
            raise ValueError(f"LQER node '{self.name}': rank must be >= 1, got {rank}")
        self.lqer_error_matrix = np.asarray(error_matrix, dtype=np.float64)
        self.lqer_rank = int(rank)
        self.lqer_factorizer = factorizer if factorizer is not None else svd_factorizer
        self.metadata['lqer_rank'] = self.lqer_rank

    def _factorize_checked(self, error_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        A, B = self.lqer_factorizer(error_2d, self.lqer_rank)
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        if (A.ndim != 2 or B.ndim != 2
                or A.shape[0] != error_2d.shape[0]
                or B.shape[1] != error_2d.shape[1]
                or A.shape[1] != B.shape[0]):
            raise ValueError(
                f"LQER node '{self.name}': factorizer returned incompatible "
                f"factors A{A.shape} @ B{B.shape} for error matrix "
                f"{error_2d.shape}"
            )
        return A, B

    def _register_lqer_params(self, ir_graph, A: np.ndarray, B: np.ndarray) -> Tuple[str, str]:
        a_name = f"{self.name}_lqer_A"
        b_name = f"{self.name}_lqer_B"
        ir_graph.add_parameter(a_name, A)
        ir_graph.add_parameter(b_name, B)
        return a_name, b_name


class _LQERLinearBranch(_LQERBranchBase):
    """Declares the x @ A -> @ B -> add diamond for linear layers."""

    def get_parallel_branch(self, ir_graph) -> Optional[Tuple[List[IRNode], IRNode]]:
        in_f = int(self.metadata['in_features'])
        out_f = int(self.metadata['out_features'])

        E = self.lqer_error_matrix
        if E.shape != (in_f, out_f):
            raise ValueError(
                f"LQER node '{self.name}': error matrix shape {E.shape} does "
                f"not match weight layout [in_features, out_features] = "
                f"({in_f}, {out_f})"
            )

        A, B = self._factorize_checked(E)
        r = int(A.shape[1])
        a_name, b_name = self._register_lqer_params(ir_graph, A, B)

        down_shape = None
        if self.inputs and self.inputs[0].output_shape:
            src_shape = list(self.inputs[0].output_shape)
            down_shape = tuple(src_shape[:-1] + [r])

        down = LQERMatmulNode(
            name=f"{self.name}_lqer_down",
            in_features=in_f,
            out_features=r,
            weight_name=a_name,
            role='down',
            corrected_layer=self.name,
            rank=r,
            output_shape=down_shape,
        )
        up = LQERMatmulNode(
            name=f"{self.name}_lqer_up",
            in_features=r,
            out_features=out_f,
            weight_name=b_name,
            role='up',
            corrected_layer=self.name,
            rank=r,
            output_shape=self.output_shape,
        )
        join = LQERAddNode(
            name=f"{self.name}_lqer_add",
            corrected_layer=self.name,
            output_shape=self.output_shape,
        )
        return [down, up], join


class _LQERConvBranch(_LQERBranchBase):
    """Declares the conv(x, A) -> 1x1 conv(B) -> add diamond for conv2d layers."""

    def get_parallel_branch(self, ir_graph) -> Optional[Tuple[List[IRNode], IRNode]]:
        if self.op_type != 'conv2d':
            raise ValueError(
                f"LQER conv correction supports conv2d only, got "
                f"'{self.op_type}' for node '{self.name}'"
            )
        groups = int(self.metadata.get('groups', 1))
        if groups != 1:
            raise ValueError(
                f"LQER node '{self.name}': grouped/depthwise conv is not "
                f"supported (groups={groups})"
            )

        kernel_size = self.metadata['kernel_size']
        k_h, k_w = (int(kernel_size[0]), int(kernel_size[1])) \
            if isinstance(kernel_size, (tuple, list)) else (int(kernel_size), int(kernel_size))
        in_c = int(self.metadata['in_channels'])
        out_c = int(self.metadata['out_channels'])
        K = k_h * k_w * in_c

        # Accept E in the stored HWIO weight layout [kh, kw, in_c, out_c]
        # or pre-flattened [kh*kw*in_c, out_c].
        E = self.lqer_error_matrix
        if E.ndim == 4:
            if E.shape != (k_h, k_w, in_c, out_c):
                raise ValueError(
                    f"LQER node '{self.name}': 4D error matrix shape {E.shape} "
                    f"does not match HWIO weight layout "
                    f"({k_h}, {k_w}, {in_c}, {out_c})"
                )
            E2 = E.reshape(K, out_c)
        elif E.ndim == 2 and E.shape == (K, out_c):
            E2 = E
        else:
            raise ValueError(
                f"LQER node '{self.name}': error matrix shape {E.shape} must "
                f"be HWIO ({k_h}, {k_w}, {in_c}, {out_c}) or flattened "
                f"({K}, {out_c})"
            )

        A, B = self._factorize_checked(E2)
        r = int(A.shape[1])
        a_name, b_name = self._register_lqer_params(
            ir_graph,
            A.reshape(k_h, k_w, in_c, r),
            B.reshape(1, 1, r, out_c),
        )

        down_shape = None
        if self.output_shape and len(self.output_shape) == 4:
            _, _, h_out, w_out = self.output_shape
            down_shape = (1, r, int(h_out), int(w_out))

        down = LQERConvMatmulNode(
            name=f"{self.name}_lqer_down",
            kernel_size=(k_h, k_w),
            stride=self.metadata['stride'],
            padding=self.metadata['padding'],
            in_channels=in_c,
            out_channels=r,
            weight_name=a_name,
            role='down',
            corrected_layer=self.name,
            rank=r,
            output_shape=down_shape,
        )
        up = LQERConvMatmulNode(
            name=f"{self.name}_lqer_up",
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            in_channels=r,
            out_channels=out_c,
            weight_name=b_name,
            role='up',
            corrected_layer=self.name,
            rank=r,
            output_shape=self.output_shape,
        )
        join = LQERAddNode(
            name=f"{self.name}_lqer_add",
            corrected_layer=self.name,
            output_shape=self.output_shape,
        )
        return [down, up], join


# ---------------------------------------------------------------------------
# LQER-corrected quantized layer nodes
# ---------------------------------------------------------------------------

class LQERDynamicQuantLinearNode(_LQERLinearBranch, DynamicQuantLinearNode):
    """DynamicQuantLinearNode + LQER error-correction branch."""

    def __init__(
        self,
        original_node: IRNode,
        dtype: str,
        weight_scale: float,
        error_matrix: np.ndarray,
        rank: int,
        offset: int = 0,
        factorizer: Optional[Callable] = None,
    ):
        DynamicQuantLinearNode.__init__(
            self, original_node=original_node, dtype=dtype,
            weight_scale=weight_scale, offset=offset,
        )
        self._init_lqer(error_matrix, rank, factorizer)


class LQERStaticQuantLinearNode(_LQERLinearBranch, StaticQuantLinearNode):
    """StaticQuantLinearNode + LQER error-correction branch (joins after dequant)."""

    def __init__(
        self,
        original_node: IRNode,
        dtype: str,
        input_scale: float,
        weight_scale: float,
        output_scale: float,
        error_matrix: np.ndarray,
        rank: int,
        input_offset: int = 0,
        weight_offset: int = 0,
        output_offset: int = 0,
        factorizer: Optional[Callable] = None,
    ):
        StaticQuantLinearNode.__init__(
            self, original_node=original_node, dtype=dtype,
            input_scale=input_scale, weight_scale=weight_scale,
            output_scale=output_scale, input_offset=input_offset,
            weight_offset=weight_offset, output_offset=output_offset,
        )
        self._init_lqer(error_matrix, rank, factorizer)


class LQERDynamicQuantConv2dNode(_LQERConvBranch, DynamicQuantConv2dNode):
    """DynamicQuantConv2dNode + LQER error-correction branch."""

    def __init__(
        self,
        original_node: IRNode,
        dtype: str,
        weight_scale: float,
        error_matrix: np.ndarray,
        rank: int,
        offset: int = 0,
        factorizer: Optional[Callable] = None,
    ):
        DynamicQuantConv2dNode.__init__(
            self, original_node=original_node, dtype=dtype,
            weight_scale=weight_scale, offset=offset,
        )
        self._init_lqer(error_matrix, rank, factorizer)


class LQERStaticQuantConv2dNode(_LQERConvBranch, StaticQuantConv2dNode):
    """StaticQuantConv2dNode + LQER error-correction branch (joins after dequant)."""

    def __init__(
        self,
        original_node: IRNode,
        dtype: str,
        input_scale: float,
        weight_scale: float,
        output_scale: float,
        error_matrix: np.ndarray,
        rank: int,
        input_offset: int = 0,
        weight_offset: int = 0,
        output_offset: int = 0,
        factorizer: Optional[Callable] = None,
    ):
        StaticQuantConv2dNode.__init__(
            self, original_node=original_node, dtype=dtype,
            input_scale=input_scale, weight_scale=weight_scale,
            output_scale=output_scale, input_offset=input_offset,
            weight_offset=weight_offset, output_offset=output_offset,
        )
        self._init_lqer(error_matrix, rank, factorizer)
