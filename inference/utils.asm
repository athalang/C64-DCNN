#importonce

.const JAM	= $02	// Jam opcode

.macro str_immediate_u16_u16 (imm, addr) {
		lda #imm
		sta addr
		lda #(imm >> 8) // Get first byte
		sta addr+1
}

.macro str_absolute_u16_u16 (lo, addr) {
		lda lo
		sta addr
		lda lo+1
		sta addr+1
}

.macro add_immediate_u16 (n, lo, hi, res) {
		clc
		lda n
		adc #lo
		sta res
		lda n+1
		adc #hi
		sta res+1
		bvc !+
		.byte JAM // Overflow error
!:		nop
}

.macro cmp_immediate_u16 (imm, m, tmp) {
		lda #imm
		sec
		sbc m
		php
		lda #(imm >> 8)
		sbc m+1
		php
		pla
		sta tmp
		pla
		and #%00000010
		ora #%11111101
		and tmp
		pha
		plp
}

.macro cmp_absolute_u16 (n, m, tmp) {
		lda n
		sec
		sbc m
		php
		lda n+1
		sbc m+1
		php
		pla
		sta tmp
		pla
		and #%00000010
		ora #%11111101
		and tmp
		pha
		plp
}

* = $2000

model_params:		.import binary "../model.bin"

// Dense input shape
.const DENSE_ROWS	= 28
.const DENSE_COLS	= 28
.const DENSE_CELLS	= DENSE_ROWS * DENSE_COLS

dense_image:		.import binary "../three.bin"

// Addresses of input matrix properties
.const INPUT_TYPE	= input_image		// 1 byte
.const INPUT_ROWS	= input_image + 1	// 1 byte
.const INPUT_COLS	= input_image + 2	// 1 byte
.const INPUT_NNZ	= input_image + 3	// 2 bytes
.const INPUT_ROW_ADDR	= input_image + 5	// 2 bytes
.const INPUT_COL_ADDR	= input_image + 7	// 2 bytes
.const INPUT_VAL_ADDR	= input_image + 9	// 2 bytes
.const INPUT_ROW_ARRAY	= INPUT_VAL_ADDR + 2
.const INPUT_COL_ARRAY	= INPUT_ROW_ARRAY + DENSE_ROWS * 2
.const INPUT_VAL_ARRAY	= INPUT_COL_ARRAY + DENSE_CELLS

input_image:
// Set matrix type to CSR, row and col are 8 bit
.byte %01100001
.byte DENSE_ROWS	// Set rows
.byte DENSE_COLS	// Set cols
.word 0			// Clear nnz
.word INPUT_ROW_ARRAY	// Set row ptr address
.word INPUT_COL_ARRAY	// Set col index address
.word INPUT_VAL_ARRAY	// Set value address
// Add enough space for the largest possible "sparse matrix"
.fill DENSE_CELLS * 3 + DENSE_ROWS * 2, 0