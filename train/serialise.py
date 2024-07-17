import torch

from fxpmath import Fxp

from optimum.quanto import safe_load, requantize, QConv2d, QLinear

from net import Net
from classes import *

def populateZeroMatrix(tensor: torch.Tensor) -> ZeroMatrix:
    matrix = ZeroMatrix(
        row_i=tensor.shape[0],
        col_i=tensor.shape[1],
    )

    return matrix

def populateCSRMatrix(sparse: torch.Tensor) -> CSRMatrix:
    row_ptrs = [row.item() for row in sparse.crow_indices()[:-1]]
    col_indices = [col.item() for col in sparse.col_indices()]
    values = [val.item() for val in sparse.values()]

    matrix = CSRMatrix(
        row_i=sparse.shape[0],
        col_i=sparse.shape[1],
        # last element of crow_indices is nnz
        nnz_i=sparse.crow_indices()[-1].item(),
        row_ptr_a=row_ptrs,
        col_index_a=col_indices,
        value_a=values,
    )

    return matrix

def populateConv(qconv: QConv2d) -> Conv:
    nested_matrices: list[list[Matrix]] = []
    nested_scales: list[list[c_uint16]] = []

    for out_i in range(0, qconv.out_channels):
        matrices: list[Matrix] = []
        scales: list[c_uint16] = []

        for in_i in range(0, qconv.in_channels):
            filter_weights = qconv.weight._data[out_i][in_i].to_sparse_csr()
            filter_scale = torch.flatten(qconv.weight._scale)[out_i].item()

            if filter_scale == 0:
                matrix = populateZeroMatrix(filter_weights)
            else:
                matrix = populateCSRMatrix(filter_weights)

            matrices.append(matrix)
            scales.append(Fxp(filter_scale, dtype='fxp-u16/15'))

        nested_matrices.append(matrices)
        nested_scales.append(scales)

    return Conv(
        in_i=qconv.in_channels,
        out_i=qconv.out_channels,
        kernel_i=qconv.kernel_size[0],
        scale_aa=nested_scales,
        matrix_aa=nested_matrices,
    )

def populateFC(qlin: QLinear) -> FullConn:
    biases = [Fxp(bias.item(), dtype='fxp-u16/15') for bias in qlin.bias]

    filter_scales = [Fxp(scale.item(), dtype='fxp-u16/15') for scale in torch.flatten(qlin.weight._scale)]
    filter_weights = qlin.weight._data.to_sparse_csr()

    matrix = populateCSRMatrix(filter_weights)

    return FullConn(
        in_i=qlin.in_features,
        out_i=qlin.out_features,
        bias_i=len(biases),
        bias_a=biases,
        scale_a=filter_scales,
        matrix_o=matrix,
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
        populateFC(model.fc1),
        ReLu(),
        populateFC(model.fc2),
        ReLu(),
        populateFC(model.fc3),
        ArgMax(),
    ]

    structure = Model(len(forwards), forwards)

    print(forwards)