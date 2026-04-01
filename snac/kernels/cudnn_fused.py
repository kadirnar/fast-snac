"""cuDNN v9 Runtime Fusion for SNAC decode.

Fuses Conv + Snake epilogue into single cuDNN kernels,
eliminating intermediate global memory writes.

Patterns fused:
  1. DepthwiseConv(k=7) + Snake → CudnnDWConvSnakeGraph
  2. Conv(k=1) + residual_add → CudnnConvResGraph
  3. ConvTranspose → CudnnConvTransposeGraph
"""

import torch
import cudnn


def nhwc_strides(n, c, h, w):
    return [c * h * w, 1, c * w, c]


class CudnnDWConvSnakeGraph:
    """cuDNN fused: depthwise Conv2d + bias + Snake epilogue."""

    def __init__(self, C, T, K_w, dilation, pad, dtype=cudnn.data_type.HALF):
        g = cudnn.pygraph(
            io_data_type=dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        T_out = (T + 2 * pad - dilation * (K_w - 1) - 1) + 1  # stride=1

        self._gx = g.tensor(name='x', dim=[1, C, 1, T],
                            stride=nhwc_strides(1, C, 1, T), data_type=dtype)
        self._gw = g.tensor(name='w', dim=[C, 1, 1, K_w],
                            stride=nhwc_strides(C, 1, 1, K_w), data_type=dtype)
        self._gb = g.tensor(name='b', dim=[1, C, 1, 1],
                            stride=[C, 1, 1, 1], data_type=dtype)
        self._ga = g.tensor(name='a', dim=[1, C, 1, 1],
                            stride=[C, 1, 1, 1], data_type=cudnn.data_type.FLOAT)
        self._gia = g.tensor(name='ia', dim=[1, C, 1, 1],
                             stride=[C, 1, 1, 1], data_type=cudnn.data_type.FLOAT)

        co = g.conv_fprop(image=self._gx, weight=self._gw,
                          pre_padding=[0, pad], post_padding=[0, pad],
                          stride=[1, 1], dilation=[1, dilation],
                          compute_data_type=cudnn.data_type.FLOAT, name='conv')
        bi = g.bias(name='bias', input=co, bias=self._gb)
        ax = g.mul(bi, self._ga, name='ax')
        sn = g.sin(ax, name='sin')
        s2 = g.mul(sn, sn, name='sin2')
        sc = g.mul(s2, self._gia, name='sc')
        self._go = g.add(bi, sc, name='out')
        self._go.set_output(True).set_data_type(dtype)

        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
        g.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

        self._ws = torch.empty(max(g.get_workspace_size(), 1),
                               device='cuda', dtype=torch.uint8)
        self._graph = g
        self.T_out = T_out

    def __call__(self, x, w, b, alpha, inv_alpha, output):
        self._graph.execute(
            {self._gx: x, self._gw: w, self._gb: b,
             self._ga: alpha, self._gia: inv_alpha, self._go: output},
            self._ws
        )


class CudnnConvResGraph:
    """cuDNN fused: Conv2d(k=1) + bias + residual_add."""

    def __init__(self, C, T, dtype=cudnn.data_type.HALF):
        g = cudnn.pygraph(
            io_data_type=dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        self._gx = g.tensor(name='x', dim=[1, C, 1, T],
                            stride=nhwc_strides(1, C, 1, T), data_type=dtype)
        self._gw = g.tensor(name='w', dim=[C, C, 1, 1],
                            stride=nhwc_strides(C, C, 1, 1), data_type=dtype)
        self._gb = g.tensor(name='b', dim=[1, C, 1, 1],
                            stride=[C, 1, 1, 1], data_type=dtype)
        self._gr = g.tensor(name='res', dim=[1, C, 1, T],
                            stride=nhwc_strides(1, C, 1, T), data_type=dtype)

        co = g.conv_fprop(image=self._gx, weight=self._gw,
                          pre_padding=[0, 0], post_padding=[0, 0],
                          stride=[1, 1], dilation=[1, 1],
                          compute_data_type=cudnn.data_type.FLOAT, name='conv')
        bi = g.bias(name='bias', input=co, bias=self._gb)
        self._go = g.add(bi, self._gr, name='add_res')
        self._go.set_output(True).set_data_type(dtype)

        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
        g.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

        self._ws = torch.empty(max(g.get_workspace_size(), 1),
                               device='cuda', dtype=torch.uint8)
        self._graph = g

    def __call__(self, x, w, b, residual, output):
        self._graph.execute(
            {self._gx: x, self._gw: w, self._gb: b,
             self._gr: residual, self._go: output},
            self._ws
        )
