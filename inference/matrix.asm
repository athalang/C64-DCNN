#importonce
#import "utils.asm"
#import "multiply.asm"

* = $0900

matrix: {

overflow_err:	.byte JAM

// Modifies X and Y
@sparse_input:	jsr mult_init
		ldx #0

!loop:		ldy #0

!loop:		lda curr_pixel
		beq !+ // Branch when pixel is zero
		add_immediate_u16(INPUT_NNZ, 1, 0, INPUT_NNZ)

!:		add_immediate_u16(curr_pixel, 1, 0, curr_pixel)
		iny
		cpy #DENSE_COLS
		bne !loop-

		inx
		cpx #DENSE_ROWS
		bne !loop--

		.byte JAM

		rts

} // End of scope matrix