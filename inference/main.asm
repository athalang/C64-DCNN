layer_iter_max	= $08
curr_layer	= $09 ; 2 bytes

!ifndef lib_matrix {
lib_matrix	!src "inference/matrix.asm"
}
!ifndef lib_multiply {
lib_multiply	!src "inference/multiply.asm"
}

* = $0801

dense_image	!bin "../three.bin"
model_params	!bin "../model.bin"

!zone main {

.format_err	jam	; Model formatted incorrectly
.overflow_err	jam	;

		; Switch out BASIC ROM
		lda #%00110110
		sta $01

		; Set up umult16 routine
		jsr mult_init

		; Jam if 0 layers in model
		lda #0
		cmp model_params
		beq .format_err

		; Store 2 * num of layers at layer_iter_max
		ldx #0
		lda model_params
		asl
		bcs .overflow_err
		sta layer_iter_max

layer_loop	; Set curr_layer to pointer
		inx ; Pointers start from 1 relative to model_params
		lda model_params,x
		sta curr_layer
		inx
		lda model_params,x
		sta curr_layer+1

		; Forward

		cpx layer_iter_max
		bne layer_loop

		rts

} ; End of zone main