import torch

from fxpmath import Fxp

from optimum.quanto import safe_load, requantize, QConv2d

from net import Net
from classes import *

def populateZeroMatrix(tensor: torch.Tensor) -> ZeroMatrix:
    matrix = ZeroMatrix(
        rows_i=tensor.shape[0],
        cols_i=tensor.shape[1],
    )

    return matrix

def populateCSRMatrix(sparse: torch.Tensor, scale: float) -> CSRMatrix:
    row_ptrs = [row.item() for row in sparse.crow_indices()[:-1]]
    col_indices = [col.item() for col in sparse.col_indices()]
    values = [val.item() for val in sparse.values()]

    matrix = CSRMatrix(
        rows_i=sparse.shape[0],
        cols_i=sparse.shape[1],
        scale_f=Fxp(scale, dtype='fxp-u16/15'),
        # last element of crow_indices is nnz
        nnz_i=sparse.crow_indices()[-1].item(),
        row_ptrs_a=row_ptrs,
        col_index_a=col_indices,
        values_a=values,
    )

    return matrix

def populateConv(qconv: QConv2d) -> Conv:
    nested: list[list[Matrix]] = []

    for out_i in range(0, qconv.out_channels):
        matrices: list[Matrix] = []

        for in_i in range(0, qconv.in_channels):
            filter_scale = torch.flatten(qconv.weight._scale)[out_i].item()
            filter_weights = qconv.weight._data[out_i][in_i].to_sparse_csr()

            if filter_scale == 0:
                matrix = populateZeroMatrix(filter_weights)
            else:
                matrix = populateCSRMatrix(filter_weights, filter_scale)

            matrices.append(matrix)

        nested.append(matrices)

    return Conv(
        in_i=qconv.in_channels,
        out_i=qconv.out_channels,
        kernel_i=qconv.kernel_size[0],
        matrix_aa=nested,
    )

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
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

    print(model.fc1.weight)
    print(model.fc1.weight._data)
    print(model.fc3.weight._data)
    print(torch.flatten(model.fc3.weight._scale))
    print(model.fc3.bias)
    print(torch.flatten(model.fc1.weight._scale))
    print(model.fc1.bias)
    print(model.fc2.weight._data)
    print(torch.flatten(model.fc2.weight._scale))
    print(model.fc2.bias)