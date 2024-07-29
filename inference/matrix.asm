dense		= $40 ; 2 bytes
rows		= $42
cols		= $43
sparse_matrix	= $44 ; 2 bytes
curr_col	= $46
curr_element	= $47 ; 2 bytes

!ifndef lib_multiply {
lib_multiply	!src "inference/multiply.asm"
}

* = $2800

!zone matrix {

.overflow_err	jam

; Input will always be 28 * 28
; Modifies X and Y
sparsify	jsr init
		ldx #0

-		ldy #0

-		; Get current element index
		; (cols * current row + current col)
		sty curr_col
		ldy cols
		jsr u8_mult
		sta curr_element+1
		clc
		lda curr_col
		adc prod_low
		sta curr_element
		bcc +
		lda #1
		adc curr_element+1
		sta curr_element+1
		bvs .overflow_err
+		ldy curr_col

		iny
		cpy $42
		bne -

		inx
		cpx $43
		bne --

		rts

} ; End of zone matrix