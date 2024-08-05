#importonce
#import "utils.asm"
#import "multiply.asm"

* = $0900

.const curr_col		= $FB
.const curr_pixel	= $FC // 2 bytes

matrix: {

overflow_err:	.byte JAM

// Modifies X and Y
@sparse_input:	jsr mult_init

		// Init zeropage counter
		lda #dense_image
		sta curr_pixel
		lda #(dense_image >> 8) // Get first byte
		sta curr_pixel+1

		ldx #0

!loop:		ldy #0

!loop:		sty curr_col
		ldy #0		// Indirect zeropage, ignore Y
		lda (curr_pixel),y
		cmp #0
		beq !+		// Branch when pixel is zero
		add_immediate_u16(INPUT_NNZ, 1, 0, INPUT_NNZ)

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