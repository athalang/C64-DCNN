from dataclasses import dataclass
from ctypes import c_int8, c_uint8, c_uint16

import torch

from fxpmath import Fxp

from optimum.quanto import safe_load, requantize, QConv2d

from net import Net

@dataclass
class Matrix:
    type_i: c_uint8
    rows_i: c_uint16
    cols_i: c_uint16
    scale_f: c_uint16 # fxp-u16/15

@dataclass(kw_only=True)
class CSRMatrix(Matrix):
    type_i: c_uint8 = 0
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

def populateConv(qconv: QConv2d) -> Conv:
    nested: list[list[Matrix]] = []
    
    for out_i in range(0, qconv.out_channels):
        matrices: list[Matrix] = []

        for in_i in range(0, qconv.in_channels):
            filter_weights = qconv.weight._data[out_i][in_i].to_sparse_csr()
            filter_scale = torch.flatten(qconv.weight._scale)[out_i].item()
            
            row_ptrs = [row.item() for row in filter_weights.crow_indices()[:-1]]
            col_indices = [col.item() for col in filter_weights.col_indices()]
            values = [val.item() for val in filter_weights.values()]
            
            matrix = CSRMatrix(
                rows_i=filter_weights.shape[0],
                cols_i=filter_weights.shape[1],
                scale_f=Fxp(filter_scale, dtype='fxp-u16/15'),
                # last element of crow_indices is nnz
                nnz_i=filter_weights.crow_indices()[-1].item(),
                row_ptrs_a=row_ptrs,
                col_index_a=col_indices,
                values_a=values,
            )
            
            matrices.append(matrix)
        
        nested.append(matrices)
    
    return Conv(
        in_i=qconv.in_channels,
        out_i=qconv.out_channels,
        kernel_i=qconv.kernel_size[0],
        matrix_aa=nested,
    )

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    torch.manual_seed(1)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = Net().to(device)

    requantize(model, safe_load('qmodel.safetensors'))
    
    forwards: list[Forward] = [
        populateConv(model.conv1),
        ReLu(),
        MaxPool(kernel_i=2),
        populateConv(model.conv2),
        ReLu(),
        MaxPool(kernel_i=2),
        Flatten(),
    ]

    structure = Model(len(forwards), forwards)
    
    print(forwards)

    print(model.fc1.weight._data)
    print(torch.flatten(model.fc1.weight._scale))
    print(model.fc1.bias)
    print(model.fc2.weight._data)
    print(torch.flatten(model.fc2.weight._scale))
    print(model.fc2.bias)
    print(model.fc3.weight._data)
    print(torch.flatten(model.fc3.weight._scale))
    print(model.fc3.bias)