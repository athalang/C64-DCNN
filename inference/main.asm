#importonce
#import "utils.asm"
#import "multiply.asm"
#import "matrix.asm"

.var layer_iter_max	= $08
.var curr_layer		= $09	// 2 bytes

* = $0801

main: {

@main_func:	// Set up multiplication tables
		jsr mult_init

		// Jam if 0 layers in model
		lda #0
		cmp model_params
		beq format_err

		// Store 2 * num of layers at layer_iter_max
		ldx #0
		lda model_params
		asl
		bcs overflow_err
		sta layer_iter_max

!:		// Set curr_layer to pointer
		inx // Pointers start from 1 relative to model_params
		lda model_params,x
		sta curr_layer
		inx
		lda model_params,x
		sta curr_layer+1

		// Forward

		cpx layer_iter_max
		bne !-

		rts

format_err:	.byte JAM	// Model formatted incorrectly
overflow_err:	.byte JAM

} // End of scope main