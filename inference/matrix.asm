#importonce
#import "defines.asm"
#import "multiply.asm"

.var dense		= $40 // 2 bytes
.var rows		= $42
.var cols		= $43
.var sparse_matrix	= $44 // 2 bytes
.var curr_col		= $46
.var curr_element	= $47 // 2 bytes

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
		clc
		lda curr_col
		adc z0
		sta curr_element
		bcc !+
		lda #1
		adc curr_element+1
		sta curr_element+1
		bvs overflow_err

!:		ldy curr_col

		iny
		cpy $42
		bne !loop-

		inx
		cpx $43
		bne !loop--

		rts

} // End of scope matrix