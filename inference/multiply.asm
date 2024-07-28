; taken from https://github.com/TobyLobster/multiply_test

; from 6502.org, by Repose: http://forum.6502.org/viewtopic.php?p=106519#p106519

; 16 bit x 16 bit unsigned multiply, 32 bit result
; Average cycles: 187.07
; 2170 bytes

; How to use:
; call jsr init, before first use
; put numbers in (x0,x1) and (y0,y1) and result is (z3, A, Y, z0)

; pointers to square tables
p_sqr_lo1    = $8b   ; 2 bytes
p_sqr_hi1    = $8d   ; 2 bytes
p_neg_sqr_lo = $8f   ; 2 bytes
p_neg_sqr_hi = $91   ; 2 bytes
p_sqr_lo2    = $93   ; 2 bytes
p_sqr_hi2    = $95   ; 2 bytes

; the inputs and outputs
x0  = p_sqr_lo1      ; multiplier, 2 bytes
x1  = p_sqr_lo2
y0  = $02            ; multiplicand, 2 bytes
y1  = $03
z0  = $04            ; product, 2 bytes + 2 registers
                     ; z1  = $05 returned in Y reg
                     ; z2  = $06 returned in A reg
z3  = $07            ;

* = $3000
!zone Mult {

; Align tables to start of page
; Note - the last byte of each table is never referenced, as a+b<=510
sqrlo
    !for i, 0, 511 {
        !byte <((i*i)/4)
    }
sqrhi
    !for i, 0, 511 {
        !byte >((i*i)/4)
    }

negsqrlo
    !for i, 0, 511 {
        !byte <(((255-i)*(255-i))/4)
    }

negsqrhi
    !for i, 0, 511 {
        !byte >(((255-i)*(255-i))/4)
    }

; Diagram of the additions
;                 y1    y0
;              x  x1    x0
;                 --------
;              x0y0h x0y0l
; +      x0y1h x0y1l
; +      x1y0h x1y0l
; +x1y1h x1y1l
; ------------------------
;     z3    z2    z1    z0

umult16
    ; set multiplier as x1
    lda x1
    sta p_sqr_hi1
    eor #$ff
    sta p_neg_sqr_lo
    sta p_neg_sqr_hi

    ; set multiplicand as y0
    ldy y0

    ; x1y0l =  low(x1*y0)
    ; x1y0h = high(x1*y0)
    sec
    lda (p_sqr_lo2),y
    sbc (p_neg_sqr_lo),y
    sta x1y0l+1
    lda (p_sqr_hi1), y
    sbc (p_neg_sqr_hi),y
    sta x1y0h+1

    ; set multiplicand as y1
    ldy y1

    ; x1y1l =  low(x1*y1)
    ; z3    = high(x1*y1)
    lda (p_sqr_lo2),y
    sbc (p_neg_sqr_lo),y
    sta x1y1l+1
    lda (p_sqr_hi1),y
    sbc (p_neg_sqr_hi),y
    sta z3

    ; set multiplier as x0
    lda x0
    sta p_sqr_hi2
    eor #$ff
    sta p_neg_sqr_lo
    sta p_neg_sqr_hi

    ; x0y1l =  low(x0*y1)
    ; X     = high(x0*y1)
    lda (p_sqr_lo1),y
    sbc (p_neg_sqr_lo),y
    sta x0y1l+1
    lda (p_sqr_hi2),y
    sbc (p_neg_sqr_hi),y
    tax

    ; set multiplicand as y0
    ldy y0

    ; z0    =  low(x0*y0)
    ; A     = high(x0*y0)
    lda (p_sqr_lo1),y
    sbc (p_neg_sqr_lo),y
    sta z0
    lda (p_sqr_hi2),y
    sbc (p_neg_sqr_hi),y

    clc
do_adds
    ; add the first two numbers of column 1
x0y1l
    adc #0      ; x0y0h + x0y1l
    tay

    ; continue to first two numbers of column 2
    txa
x1y0h
    adc #0      ; x0y1h + x1y0h
    tax         ; X=z2 so far
    bcc +
    inc z3      ; column 3
    clc

    ; add last number of column 1
+
    tya
x1y0l
    adc #0      ; + x1y0l
    tay         ; Y=z1

    ; add last number of column 2
    txa
x1y1l
    adc #0      ; + x1y1l
    bcc fin     ; A=z2
    inc z3      ; column 3
fin
    rts

; Initialises both
init
    lda #>sqrlo
    sta p_sqr_lo2+1
    sta p_sqr_lo1+1

    lda #>sqrhi
    sta p_sqr_hi1+1
    sta p_sqr_hi2+1

    lda #>negsqrlo
    sta p_neg_sqr_lo+1

    lda #>negsqrhi
    sta p_neg_sqr_hi+1

    lda #>squaretable1_lsb          ; high byte (#2 in this instance)
    sta lmul0+1
    lda #>squaretable1_msb          ; high byte (#4 in this instance)
    sta lmul1+1
    rts

; from TobyLobster, based on Nick Jameson's 3D Demo for the BBC Micro (1994), https://github.com/simondotm/bbc-micro-3d/blob/master/source/culling.asm

; 8 bit x 8bit unsigned multiply, 16 bit result
; Average cycles: 45.49
; 1580 bytes

result          = $04   ; 2 bytes

lmul0           = $06   ; pointer into table 1 low
lmul1           = $08   ; pointer into table 1 high
prod_low        = $0a

; table1 = n*n/4, where n=0..510
; table2 = 0 if n=0 else (256-n)*(n-256)/4, where n=0..255

; Table 1 must be aligned to start of a page

squaretable1_lsb
    !for i, 0, 510 {
        !byte <((i*i)/4)
    }
    !byte 0     ; unused, needed for alignment of next table

squaretable1_msb
    !for i, 0, 510 {
        !byte >((i*i)/4)
    }
    !byte 0     ; unused, needed for alignment of next table

; Table 2 should be aligned to the start of a page for speed
squaretable2_lsb
    !byte 0
    !for i, 1, 255 {
        !byte <(((256-i)*(256-i))/4-1)
    }
squaretable2_msb
    !byte 0
    !for i, 1, 255 {
        !byte >(((256-i)*(256-i))/4-1)
    }

; Unsigned multiplication of two 8-bit terms is computed as:
;   if Y >= X:
;       r = table1[X+Y] - table1[Y-X]
;   else:
;       r = table1[X+Y] - table2[Y-X]
; where r is a 16-bit unsigned result
; and 'Y-X' is calculated as a single byte value (0 to 255)

; 8 bit x 8bit unsigned multiply, 16 bit result

; On Entry:
;  X: multiplier
;  Y: multiplicand
; On Exit:
;  (prod_low, A): product
umult8
    stx lmul0
    stx lmul1
    tya
    sec
    sbc lmul0
    tax
    lda (lmul0),Y
    bcc +
    sbc squaretable1_lsb,X
    sta prod_low
    lda (lmul1),Y
    sbc squaretable1_msb,X
    rts
+
    sbc squaretable2_lsb,X
    sta prod_low
    lda (lmul1),Y
    sbc squaretable2_msb,X
    rts
}