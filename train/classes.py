from dataclasses import dataclass
from ctypes import c_int8, c_uint8, c_uint16

@dataclass
class Matrix:
    type_i: c_uint8
    row_i: c_uint16
    col_i: c_uint16

@dataclass(kw_only=True)
class ZeroMatrix(Matrix):
    type_i: c_uint8 = 0

@dataclass(kw_only=True)
class CSRMatrix(Matrix):
    type_i: c_uint8 = 1
    nnz_i: c_uint16
    row_ptr_i: c_uint16
    row_ptr_a: list[c_uint16] # row_ptr_i bytes
    col_index_a: list[c_uint16] # nnz_i * 2 bytes
    value_a: list[c_int8] # nnz_i bytes

@dataclass
class Forward:
    type_i: c_uint8

@dataclass(kw_only=True)
class ReLu(Forward):
    type_i: c_uint8 = 0

@dataclass(kw_only=True)
class ArgMax(Forward):
    type_i: c_uint8 = 1

@dataclass
class Layer(Forward):
    in_i: c_uint16
    out_i: c_uint16

@dataclass(kw_only=True)
class FullConn(Layer):
    type_i: c_uint8 = 2
    bias_a: list[c_uint16] # list[fxp-u16/15], out_i * 2 bytes
    scale_a: list[c_uint16] # list[fxp-u16/15], out_i * 2 bytes
    matrix_o: Matrix

@dataclass(kw_only=True)
class Conv(Layer):
    type_i: c_uint8 = 3
    kernel_i: c_uint8
    padding_i: c_uint8

    # list of out_i lists, with in_i matrices each
    scale_aa: list[list[c_uint16]] # list[list[fxp-u16/15]]
    matrix_aa: list[list[Matrix]]

@dataclass(kw_only=True)
class MaxPool(Forward):
    type_i: c_uint8 = 4
    kernel_i: c_uint8

@dataclass(kw_only=True)
class Flatten(Forward):
    type_i: c_uint8 = 5

@dataclass
class Model:
    forwards_i: c_uint8
    forwards_a: list[Forward]