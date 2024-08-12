#importonce
#import "utils.asm"
#import "memory.asm"
#import "multiply.asm"

* = $0900

.const curr_col		= $2B
.const curr_pixel	= $2C	// 2 bytes
.const curr_row_ptr	= $2E	// 2 bytes
.const curr_col_index	= $30	// 2 bytes
.const curr_value	= $32	// 2 bytes
.const start_of_row	= $34
.const prev_row_ptr_val	= $35	// 2 bytes

matrix: {

// Modifies X and Y
@sparse_input:	jsr heap_init
		jsr mult_init

		// Init zeropage counter
		str_immediate_u16_u16(dense_image, curr_pixel)
		str_immediate_u16_u16(INPUT_ROW_ARRAY, curr_row_ptr)
		str_immediate_u16_u16(INPUT_COL_ARRAY, curr_col_index)
		str_immediate_u16_u16(INPUT_VAL_ARRAY, curr_value)
		str_immediate_u16_u16(0, prev_row_ptr_val)

		ldx #0

!loop:		ldy #0
		sty start_of_row

!loop:		sty curr_col
		ldy #0		// Indirect zeropage, ignore Y
		lda (curr_pixel),y

		beq !++		// Ignore when pixel is zero
		sta (curr_value),y	// Current pixel in acc
		add_immediate_u16(curr_value, 1, 0, curr_value)
		lda curr_col
		sta (curr_col_index),y
		add_immediate_u16(curr_col_index, 1, 0, curr_col_index)

		lda start_of_row
		bne !+
		lda #1
		sta start_of_row
		lda INPUT_NNZ
		sta (curr_row_ptr),y
		lda INPUT_NNZ+1
		iny
		sta (curr_row_ptr),y
		str_absolute_2u8_u16(INPUT_NNZ, prev_row_ptr_val)
		add_immediate_u16(curr_row_ptr, 2, 0, curr_row_ptr)

!:		add_immediate_u16(INPUT_NNZ, 1, 0, INPUT_NNZ)

!:		add_immediate_u16(curr_pixel, 1, 0, curr_pixel)
		ldy curr_col
		iny
		cpy #DENSE_COLS
		bne !loop-

		lda start_of_row
		bne !+
		// Place previous rowptr value
		ldy #0
		lda prev_row_ptr_val
		sta (curr_row_ptr),y
		lda prev_row_ptr_val+1
		iny
		sta (curr_row_ptr),y
		add_immediate_u16(curr_row_ptr, 2, 0, curr_row_ptr)

!:		inx
		cpx #DENSE_ROWS
		beq !+
		jmp !loop-- // loop-- too far away for relative mode

!:		.byte JAM

		rts

} // End of scope matrix