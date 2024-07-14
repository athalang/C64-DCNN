from dataclasses import dataclass
from ctypes import c_int8, c_uint8, c_uint16

import torch

from optimum.quanto import safe_load, requantize

from net import Net

@dataclass
class Matrix:
    type_i: c_uint8
    rows_i: c_uint16
    cols_i: c_uint16
    scale_f: c_int8 # fxp-u8/7 (twos complement)

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
    matrix_a: list[Matrix] # length is out_i

@dataclass(kw_only=True)
class FullConn(Layer):
    type_i: c_uint8 = 2
    biases_i: c_uint16
    biases_a: list[c_int8] # list[fxp-u8/7] (twos complement)

@dataclass(kw_only=True)
class Conv(Layer):
    type_i: c_uint8 = 3
    kernel_i: c_uint8
    padding_i: c_uint8 = 0

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
        Conv(in_i=1, out_i=6, kernel_i=5, matrix_a=[]),
    ]
    
    structure = Model(len(forwards), forwards)
    
    print(torch.flatten(model.conv1.weight._data, start_dim=1))
    print(torch.flatten(model.conv1.weight._scale))
    print(torch.flatten(model.conv2.weight._data, start_dim=1))
    print(torch.flatten(model.conv2.weight._scale))
    print(model.fc1.weight._data)
    print(torch.flatten(model.fc1.weight._scale))
    print(model.fc1.bias)
    print(model.fc2.weight._data)
    print(torch.flatten(model.fc2.weight._scale))
    print(model.fc2.bias)
    print(model.fc3.weight._data)
    print(torch.flatten(model.fc3.weight._scale))
    print(model.fc3.bias)