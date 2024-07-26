from dataclasses import dataclass
from ctypes import c_int8, c_uint8, c_uint16

@dataclass
class Matrix:
    type_i: c_uint8

@dataclass(kw_only=True)
class ZeroMatrix(Matrix):
    type_i: c_uint8 = 0
    row_i: c_uint16
    col_i: c_uint16

@dataclass(kw_only=True)
class CSRMatrix(Matrix):
    ## bitfield
    # 7th bit determines if nnz is uint8 or uint16
    # 6th and 5th bits determine if rows and cols are uint8 or uint16
    # 4th bit determines if row_ptrs are uint8 or uint16
    # 0th bit determines data struct, will be 1
    type_i: c_uint8 = 1

    nnz_i: int
    row_i: int
    col_i: int
    row_ptr_a: list[int] # row_i * int bytes
    col_index_a: list[int] # nnz_i * int bytes
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
    scale_a: list[c_uint16] # list[fxp-u16/15]
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