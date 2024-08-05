#importonce
#import "utils.asm"
#import "multiply.asm"

* = $0900

.const curr_col		= $2B
.const curr_pixel	= $2C // 2 bytes
.const curr_row_ptr	= $2E // 2 bytes
.const curr_col_index	= $30 // 2 bytes
.const curr_value	= $32 // 2 bytes

matrix: {

overflow_err:	.byte JAM

// Modifies X and Y
@sparse_input:	jsr mult_init

		// Init zeropage counter
		str_immediate_u16_u16(dense_image, curr_pixel)
		str_immediate_u16_u16(INPUT_ROW_ARRAY, curr_row_ptr)
		str_immediate_u16_u16(INPUT_COL_ARRAY, curr_col_index)
		str_immediate_u16_u16(INPUT_VAL_ARRAY, curr_value)
		str_immediate_u16_u16(0, curr_element)

		ldx #0

!loop:		ldy #0

!loop:		sty curr_col
		ldy #0		// Indirect zeropage, ignore Y
		lda (curr_pixel),y

		beq !+		// Ignore when pixel is zero
		sta (curr_value),y	// Current pixel in acc
		add_immediate_u16(curr_value, 1, 0, curr_value)
		add_immediate_u16(INPUT_NNZ, 1, 0, INPUT_NNZ)
		add_immediate_u16(curr_element, 1, 0, curr_element)
		lda curr_col
		sta (curr_col_index),y
		add_immediate_u16(curr_col_index, 1, 0, curr_col_index)

!:		add_immediate_u16(curr_pixel, 1, 0, curr_pixel)
		ldy curr_col
		iny
		cpy #DENSE_COLS
		bne !loop-

		inx
		cpx #DENSE_ROWS
		bne !loop--

		.byte JAM

		rts

} // End of scope matrix