#importonce

.const JAM	= $02	// Jam opcode

.macro add_immediate_u16 (n, lo, hi, res) {
		clc
		lda n
		adc #lo
		sta res
		lda n+1
		adc #hi
		sta res+1
}

* = $2000

model_params:		.import binary "../model.bin"

curr_pixel:		.word dense_image

dense_image:		.import binary "../three.bin"

// Dense input shape
.const DENSE_ROWS	= 28
.const DENSE_COLS	= 28
.const DENSE_CELLS	= DENSE_ROWS * DENSE_COLS

input_image:
.byte %00000001			// Set matrix type to CSR
.word DENSE_ROWS		// Set rows
.word DENSE_COLS		// Set cols
// Add enough space for the largest possible "sparse matrix"
// i.e. when all cells are nonzero, will need 3 * nnz
.fill DENSE_CELLS * 3 + 10, 0

// Addresses of input matrix properties
.const INPUT_TYPE	= input_image		// 1 byte
.const INPUT_ROWS	= input_image + 1	// 2 bytes
.const INPUT_COLS	= input_image + 3	// 2 bytes
.const INPUT_NNZ	= input_image + 5	// 2 bytes
.const INPUT_ROWPTRS	= input_image + 7	// 2 bytes
.const INPUT_COLINDEX	= input_image + 9	// 2 bytes
.const INPUT_VALUES	= input_image + 11	// 2 bytes