from dataclasses import dataclass
from ctypes import c_int8, c_uint8, c_uint16

@dataclass
class Matrix:
    type_i: c_uint8
    rows_i: c_uint16
    cols_i: c_uint16

@dataclass(kw_only=True)
class ZeroMatrix(Matrix):
    type_i: c_uint8 = 0

@dataclass(kw_only=True)
class CSRMatrix(Matrix):
    type_i: c_uint8 = 1
    scale_f: c_uint16 # fxp-u16/15
    nnz_i: c_uint16 # number of non-zero elements
    row_ptrs_a: list[c_uint16] # length is nnz_i
    col_index_a: list[c_uint16] # length is nnz_i
    values_a: list[c_int8] # length is nnz_i

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
    biases_i: c_uint16
    biases_a: list[c_uint16] # list[fxp-u16/15]
    matrix_a: list[Matrix] # length is out_i

@dataclass(kw_only=True)
class Conv(Layer):
    type_i: c_uint8 = 3
    kernel_i: c_uint8
    padding_i: c_uint8 = 0
    # list of out_i lists, with in_i matrices each
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