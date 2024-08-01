import torch
from optimum.quanto import safe_load, requantize

from net import Net
from layers import *

def place16BitWord(word: int, bytes: bytearray, pointer: int):
    # little endian, highest byte first
    bytes[pointer] = word & 0xff
    bytes[pointer + 1] = (word >> 8) & 0xff

def append16BitWord(word: int, bytes: bytearray):
    # little endian, highest byte first
    bytes.append(word & 0xff)
    bytes.append((word >> 8) & 0xff)

def appendWithFlag(value: int, bitfield: int, flag: int, bytes: bytearray):
    if bitfield & flag == flag:
        bytes.append(value)
    else:
        append16BitWord(value, bytes)

def matrixBytes(matrix: Matrix, out: bytearray):
    out.append(matrix.type_i)
    appendWithFlag(matrix.row_i, matrix.type_i, 0b01000000, out)

    appendWithFlag(matrix.col_i, matrix.type_i, 0b00100000, out)

    if isinstance(matrix, CSRMatrix):
        appendWithFlag(matrix.nnz_i, matrix.type_i, 0b10000000, out)

        # Add placeholder pointer bytes
        pointers = len(out)
        out.extend([0 for i in range(3 * 2)])

        place16BitWord(len(out), out, pointers)
        for row_ptr in matrix.row_ptr_a:
            appendWithFlag(row_ptr, matrix.type_i, 0b00010000, out)

        place16BitWord(len(out), out, pointers + 2)
        for col_index in matrix.col_index_a:
            appendWithFlag(col_index, matrix.type_i, 0b00100000, out)

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

    layers: list[Layer] = [
        Conv.fromQconv2d(model.conv1), ReLu(), MaxPool(kernel_i=2),
        Conv.fromQconv2d(model.conv2), ReLu(), MaxPool(kernel_i=2),
        Flatten(),
        FullConn.fromQlinear(model.fc1), ReLu(),
        FullConn.fromQlinear(model.fc2), ReLu(),
        FullConn.fromQlinear(model.fc3),
        ArgMax(),
    ]

    out = bytearray() # See model.hexpat for file format description
    out.append(len(layers))

    # Add placeholder pointer bytes
    out.extend([0 for i in range(len(layers) * 2)])

    for current_layer in range(len(layers)):
        place16BitWord(len(out), out, 1 + current_layer * 2)

        layer = layers[current_layer]
        out.append(layer.type_i)

        if isinstance(layer, FullConn):
            append16BitWord(layer.in_i, out)
            append16BitWord(layer.out_i, out)

            # Add placeholder pointer bytes
            pointers = len(out)
            out.extend([0 for i in range(3 * 2)])

            place16BitWord(len(out), out, pointers)
            for bias in layer.bias_a:
                append16BitWord(bias, out)

            place16BitWord(len(out), out, pointers + 2)
            for scale in layer.scale_a:
                append16BitWord(scale, out)

            place16BitWord(len(out), out, pointers + 4)
            matrixBytes(layer.matrix_o, out)
        elif isinstance(layer, Conv):
            append16BitWord(layer.in_i, out)
            append16BitWord(layer.out_i, out)
            out.extend([layer.kernel_i, layer.padding_i])

            # Add placeholder pointer bytes
            pointers = len(out)
            out.extend([0 for i in range(2 * 2)])

            place16BitWord(len(out), out, pointers)
            for scale in layer.scale_a:
                append16BitWord(scale, out)

            place16BitWord(len(out), out, pointers + 2)
            # Add placeholder pointer bytes
            matrixpointers = len(out)
            elements = len([m for row in layer.matrix_aa for m in row])
            out.extend([0 for i in range(layer.in_i * layer.out_i * 2)])

            i = 0
            for row in layer.matrix_aa:
                for matrix in row:
                    place16BitWord(len(out), out, matrixpointers + i)
                    matrixBytes(matrix, out)
                    i += 2
        elif isinstance(layer, MaxPool):
            out.append(layer.kernel_i)

    with open("model.bin", "wb") as binary_file:
        binary_file.write(out)