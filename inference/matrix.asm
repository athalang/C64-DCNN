#importonce
#import "utils.asm"
#import "multiply.asm"

.var dense		= $40 // 2 bytes
.var rows		= $42
.var cols		= $43
.var sparse_matrix	= $44 // 2 bytes
.var curr_col		= $46 // 1 byte, next byte is 0
.var curr_element	= $48 // 2 bytes
.var curr_pixel		= $4A // 2 bytes

* = $2800

matrix: {

overflow_err:	.byte JAM

// Input will always be 28 * 28
// Modifies X and Y
@sparsify:	jsr mult_init
		ldx #0

!loop:		ldy #0

!loop:		sty curr_col

		// Get current element index
		// (cols * current row + current col)
		ldy cols
		jsr u8_mult
		sta curr_element+1
		lda z0
		sta curr_element
		add_u16(curr_element, curr_col, curr_element)
		bvs overflow_err

		add_u16(curr_element, dense_image, curr_pixel)
		bvs overflow_err

		ldy curr_col
		iny
		cpy $42
		bne !loop-

		inx
		cpx $43
		bne !loop--

		rts

} // End of scope matrix