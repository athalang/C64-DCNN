#importonce
#import "utils.asm"
#import "multiply.asm"

.var curr_col		= $46 // 1 byte, next byte is 0
.var curr_element	= $48 // 2 bytes
.var curr_pixel		= $4A // 2 bytes

* = $0900

matrix: {

overflow_err:	.byte JAM

// Modifies X and Y
@sparse_input:	jsr mult_init

		ldx #0

!loop:		ldy #0

!loop:		sty curr_col

		// Get current element index
		// (cols * current row + current col)
		ldy #INPUT_COLS
		jsr u8_mult
		sta curr_element+1
		lda z0
		sta curr_element
		add_u16(curr_element, curr_col, curr_element)
		bvs overflow_err

		// Setup curr_pixel
		add_u16(curr_element, dense_image, curr_pixel)
		bvs overflow_err

		lda curr_pixel
		beq !+

!:

		ldy curr_col
		iny
		cpy $42
		bne !loop-

		inx
		cpx $43
		bne !loop--

		rts

} // End of scope matrix