import torch
from optimum.quanto import QConv2d, QLinear

from dataclasses import dataclass
from ctypes import c_uint8, c_uint16
from fxpmath import Fxp

from matrix import *

def fixedUint(float) -> int:
    return int(Fxp(float, dtype='fxp-u16/15').bin(), 2)

@dataclass
class Layer:
    type_i: c_uint8

@dataclass(kw_only=True)
class ReLu(Layer):
    type_i: c_uint8 = 0

@dataclass(kw_only=True)
class ArgMax(Layer):
    type_i: c_uint8 = 1

@dataclass(kw_only=True)
class FullConn(Layer):
    type_i: c_uint8 = 2
    in_i: c_uint16
    out_i: c_uint16
    bias_a: list[c_uint16] # list[fxp-u16/15], out_i * 2 bytes
    scale_a: list[c_uint16] # list[fxp-u16/15], out_i * 2 bytes
    matrix_o: Matrix

    def fromQlinear(qlin: QLinear):
        biases = [fixedUint(bias.item()) for bias in qlin.bias]

        filter_scales = [fixedUint(scale.item()) for scale in torch.flatten(qlin.weight._scale)]
        filter_weights = qlin.weight._data.to_sparse_csr()

        matrix = CSRMatrix.fromSparse(filter_weights)

        return FullConn(
            in_i=qlin.in_features,
            out_i=qlin.out_features,
            bias_a=biases,
            scale_a=filter_scales,
            matrix_o=matrix,
        )

@dataclass(kw_only=True)
class Conv(Layer):
    type_i: c_uint8 = 3
    in_i: c_uint16
    out_i: c_uint16
    kernel_i: c_uint8
    padding_i: c_uint8

    # list of out_i lists, with in_i matrices each
    scale_a: list[c_uint16] # list[fxp-u16/15]
    matrix_aa: list[list[Matrix]]

    @staticmethod
    def fromQconv2d(qconv: QConv2d):
        nested_matrices: list[list[Matrix]] = []
        scales: list[c_uint16] = []

        for out_i in range(0, qconv.out_channels):
            matrices: list[Matrix] = []
            filter_scale = torch.flatten(qconv.weight._scale)[out_i].item()

            for in_i in range(0, qconv.in_channels):
                filter_weights = qconv.weight._data[out_i][in_i].to_sparse_csr()

                if filter_scale == 0:
                    matrix = ZeroMatrix.fromTensor(filter_weights)
                else:
                    matrix = CSRMatrix.fromSparse(filter_weights)

                matrices.append(matrix)

            nested_matrices.append(matrices)
            scales.append(fixedUint(filter_scale))

        return Conv(
            in_i=qconv.in_channels,
            out_i=qconv.out_channels,
            kernel_i=qconv.kernel_size[0],
            padding_i=qconv.padding[0],
            scale_a=scales,
            matrix_aa=nested_matrices,
        )

@dataclass(kw_only=True)
class MaxPool(Layer):
    type_i: c_uint8 = 4
    kernel_i: c_uint8

@dataclass(kw_only=True)
class Flatten(Layer):
    type_i: c_uint8 = 5