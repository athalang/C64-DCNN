import torch

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

    @staticmethod
    def fromTensor(tensor: torch.Tensor):
        return ZeroMatrix(
        row_i=tensor.shape[0],
        col_i=tensor.shape[1],
        )

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

    @staticmethod
    def fromSparse(sparse: torch.Tensor):
        row_ptrs = [row.item() for row in sparse.crow_indices()[:-1]]
        col_indices = [col.item() for col in sparse.col_indices()]
        values = [val.item() for val in sparse.values()]

        bitfield = 1
        if sparse.crow_indices()[-1].item() <= 255:
            bitfield |= 0b10000000
        if sparse.shape[0] <= 255:
            bitfield |= 0b01000000
        if sparse.shape[1] <= 255:
            bitfield |= 0b00100000

        bitfield |= 0b00010000
        for row_ptr in row_ptrs:
            if row_ptr > 255:
                bitfield &= 0b11101111
                break

        return CSRMatrix(
            type_i=bitfield,
            row_i=sparse.shape[0],
            col_i=sparse.shape[1],
            # last element of crow_indices is nnz
            nnz_i=sparse.crow_indices()[-1].item(),
            row_ptr_a=row_ptrs,
            col_index_a=col_indices,
            value_a=values,
        )