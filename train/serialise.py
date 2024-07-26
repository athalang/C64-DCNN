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

    matrix = CSRMatrix(
        type_i=bitfield,
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
    scales: list[c_uint16] = []

    for out_i in range(0, qconv.out_channels):
        matrices: list[Matrix] = []
        filter_scale = torch.flatten(qconv.weight._scale)[out_i].item()

        for in_i in range(0, qconv.in_channels):
            filter_weights = qconv.weight._data[out_i][in_i].to_sparse_csr()

            if filter_scale == 0:
                matrix = populateZeroMatrix(filter_weights)
            else:
                matrix = populateCSRMatrix(filter_weights)

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

def populateFC(qlin: QLinear) -> FullConn:
    biases = [fixedUint(bias.item()) for bias in qlin.bias]

    filter_scales = [fixedUint(scale.item()) for scale in torch.flatten(qlin.weight._scale)]
    filter_weights = qlin.weight._data.to_sparse_csr()

    matrix = populateCSRMatrix(filter_weights)

    return FullConn(
        in_i=qlin.in_features,
        out_i=qlin.out_features,
        bias_a=biases,
        scale_a=filter_scales,
        matrix_o=matrix,
    )

def fixedUint(float) -> int:
    return int(Fxp(float, dtype='fxp-u16/15').bin(), 2)

def place16BitWord(word: int, bytes: bytearray, pointer: int):
    # little endian, highest byte first
    bytes[pointer] = word & 0xff
    bytes[pointer + 1] = (word >> 8) & 0xff

def append16BitWord(word: int, bytes: bytearray):
    # little endian, highest byte first
    bytes.append(word & 0xff)
    bytes.append((word >> 8) & 0xff)

def matrixBytes(matrix: Matrix, out: bytearray):
    out.append(matrix.type_i)
    if matrix.type_i & 0b01000000 == 0b01000000:
        out.append(matrix.row_i)
    else:
        append16BitWord(matrix.row_i, out)

    if matrix.type_i & 0b00100000 == 0b00100000:
        out.append(matrix.col_i)
    else:
        append16BitWord(matrix.col_i, out)

    if isinstance(matrix, CSRMatrix):
        if matrix.type_i & 0b10000000 == 0b10000000:
            out.append(matrix.nnz_i)
        else:
            append16BitWord(matrix.nnz_i, out)

        # Add placeholder pointer bytes
        pointers = len(out)
        out.extend([0 for i in range(3 * 2)])

        place16BitWord(len(out), out, pointers)
        for row_ptr in matrix.row_ptr_a:
            if matrix.type_i & 0b00010000 == 0b00010000:
                out.append(row_ptr)
            else:
                append16BitWord(row_ptr, out)

        place16BitWord(len(out), out, pointers + 2)
        for col_index in matrix.col_index_a:
            if matrix.type_i & 0b00100000 == 0b00100000:
                out.append(col_index)
            else:
                append16BitWord(col_index, out)

        place16BitWord(len(out), out, pointers + 4)
        for value in matrix.value_a:
            out.append(value & 0xff)

if __name__ == '__main__':
    # Suppress sparse_csr warning
    from warnings import filterwarnings
    filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

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

    out = bytearray()
    out.append(structure.forwards_i)

    # Add placeholder pointer bytes
    out.extend([0 for i in range(structure.forwards_i * 2)])

    for forward_counter in range(structure.forwards_i):
        place16BitWord(len(out), out, 1 + forward_counter * 2)

        forward = forwards[forward_counter]
        out.append(forward.type_i)

        if isinstance(forward, Layer):
            append16BitWord(forward.in_i, out)
            append16BitWord(forward.out_i, out)

            if isinstance(forward, FullConn):
                # Add placeholder pointer bytes
                pointers = len(out)
                out.extend([0 for i in range(3 * 2)])

                place16BitWord(len(out), out, pointers)
                for bias in forward.bias_a:
                    append16BitWord(bias, out)

                place16BitWord(len(out), out, pointers + 2)
                for scale in forward.scale_a:
                    append16BitWord(scale, out)

                place16BitWord(len(out), out, pointers + 4)
                matrixBytes(forward.matrix_o, out)
            elif isinstance(forward, Conv):
                out.append(forward.kernel_i)
                out.append(forward.padding_i)

                # Add placeholder pointer bytes
                pointers = len(out)
                out.extend([0 for i in range(2 * 2)])

                place16BitWord(len(out), out, pointers)
                for scale in forward.scale_a:
                    append16BitWord(scale, out)

                place16BitWord(len(out), out, pointers + 2)
                # Add placeholder pointer bytes
                matrixpointers = len(out)
                elements = len([m for row in forward.matrix_aa for m in row])
                out.extend([0 for i in range(forward.in_i * forward.out_i * 2)])

                i = 0
                for row in forward.matrix_aa:
                    for matrix in row:
                        place16BitWord(len(out), out, matrixpointers + i)
                        matrixBytes(matrix, out)
                        i += 2

        else:
            if isinstance(forward, MaxPool):
                out.append(forward.kernel_i)

    with open("model.bin", "wb") as binary_file:
        binary_file.write(out)