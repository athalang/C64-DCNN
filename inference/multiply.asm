#importonce
#import "utils.asm"

// taken from https://github.com/TobyLobster/multiply_test

// pointers to square tables
.var p_sqr_lo1		= $8b	// 2 bytes
.var p_sqr_hi1		= $8d	// 2 bytes
.var p_neg_sqr_lo	= $8f	// 2 bytes
.var p_neg_sqr_hi	= $91	// 2 bytes
.var p_sqr_lo2		= $93	// 2 bytes
.var p_sqr_hi2		= $95	// 2 bytes

// the inputs and outputs
.var x0		= p_sqr_lo1	// multiplier, 2 bytes
.var x1		= p_sqr_lo2
.var y0		= $02		// multiplicand, 2 bytes
.var y1		= $03
.var z0		= $04		// product, 2 bytes + 2 registers
				// z1  = $05 returned in Y reg
				// z2  = $06 returned in A reg
.var z3		= $07		//

* = $3000

mult: {

// Align tables to start of page
// Note - the last byte of each table is never referenced, as a+b<=510
.var i	= 0
sqrlo:	.for (i = 0; i < 512; i++) {
		.byte <((i*i)/4)
	}
sqrhi:	.for (i = 0; i < 512; i++) {
		.byte >((i*i)/4)
	}

negsqrlo:
	.for (i = 0; i < 512; i++) {
		.byte <(((255-i)*(255-i))/4)
	}

negsqrhi:
	.for (i = 0; i < 512; i++) {
		.byte >(((255-i)*(255-i))/4)
	}

// from 6502.org, by Repose: http://forum.6502.org/viewtopic.php?p=106519#p106519

// 16 bit x 16 bit unsigned multiply, 32 bit result
// Average cycles: 187.07
// 2170 bytes

// How to use:
// call jsr init, before first use
// put numbers in (x0,x1) and (y0,y1) and result is (z3, A, Y, z0)

// Diagram of the additions
//                 y1    y0
//              x  x1    x0
//                 --------
//              x0y0h x0y0l
// +      x0y1h x0y1l
// +      x1y0h x1y0l
// +x1y1h x1y1l
// ------------------------
//     z3    z2    z1    z0

@u16_mult:	lda x1			// set multiplier as x1
		sta p_sqr_hi1
		eor #$ff
		sta p_neg_sqr_lo
		sta p_neg_sqr_hi

		ldy y0			// set multiplicand as y0

		// x1y0l =  low(x1*y0)
		// x1y0h = high(x1*y0)
		sec
		lda (p_sqr_lo2),y
		sbc (p_neg_sqr_lo),y
		sta x1y0l+1
		lda (p_sqr_hi1), y
		sbc (p_neg_sqr_hi),y
		sta x1y0h+1

		ldy y1			// set multiplicand as y1

		// x1y1l =  low(x1*y1)
		// z3    = high(x1*y1)
		lda (p_sqr_lo2),y
		sbc (p_neg_sqr_lo),y
		sta x1y1l+1
		lda (p_sqr_hi1),y
		sbc (p_neg_sqr_hi),y
		sta z3

		lda x0			// set multiplier as x0
		sta p_sqr_hi2
		eor #$ff
		sta p_neg_sqr_lo
		sta p_neg_sqr_hi

		// x0y1l =  low(x0*y1)
		// X     = high(x0*y1)
		lda (p_sqr_lo1),y
		sbc (p_neg_sqr_lo),y
		sta x0y1l+1
		lda (p_sqr_hi2),y
		sbc (p_neg_sqr_hi),y
		tax

		ldy y0			// set multiplicand as y0

		// z0    =  low(x0*y0)
		// A     = high(x0*y0)
		lda (p_sqr_lo1),y
		sbc (p_neg_sqr_lo),y
		sta z0
		lda (p_sqr_hi2),y
		sbc (p_neg_sqr_hi),y

		clc
do_adds:
		// add the first two numbers of column 1
x0y1l:		adc #0	// x0y0h + x0y1l
		tay

		// continue to first two numbers of column 2
		txa

x1y0h:		adc #0      // x0y1h + x1y0h
		tax         // X=z2 so far
		bcc !+
		inc z3      // column 3
		clc

		// add last number of column 1
!:		tya

x1y0l:		adc #0      // + x1y0l
		tay         // Y=z1

		// add last number of column 2
		txa

x1y1l:		adc #0      // + x1y1l
		bcc fin     // A=z2
		inc z3      // column 3

fin:		rts

// Initialises multiplication tables
@mult_init:	lda #>sqrlo
		sta p_sqr_lo2+1
		sta p_sqr_lo1+1

		lda #>sqrhi
		sta p_sqr_hi1+1
		sta p_sqr_hi2+1

		lda #>negsqrlo
		sta p_neg_sqr_lo+1

		lda #>negsqrhi
		sta p_neg_sqr_hi+1
		rts

// from TobyLobster, based on Nick Jameson's 3D Demo for the BBC Micro (1994), https://github.com/simondotm/bbc-micro-3d/blob/master/source/culling.asm

// 8 bit x 8bit unsigned multiply, 16 bit result
// Average cycles: 45.49
// 1580 bytes

// Unsigned multiplication of two 8-bit terms is computed as:
//   if Y >= X:
//       r = table1[X+Y] - table1[Y-X]
//   else:
//       r = table1[X+Y] - table2[Y-X]
// where r is a 16-bit unsigned result
// and 'Y-X' is calculated as a single byte value (0 to 255)

// 8 bit x 8bit unsigned multiply, 16 bit result

// On Entry:
//  X: multiplier
//  Y: multiplicand
// On Exit:
//  (z0, A): product
@u8_mult:	stx p_sqr_lo1
		stx p_sqr_hi1
		tya
		sec
		sbc p_sqr_lo1
		tax
		lda (p_sqr_lo1),Y
		bcc !+
		sbc sqrlo,X
		sta z0
		lda (p_sqr_hi1),Y
		sbc sqrhi,X
		rts
!:		sbc negsqrlo,X
		sta z0
		lda (p_sqr_hi1),Y
		sbc negsqrhi,X
		rts

} // End of scope mult