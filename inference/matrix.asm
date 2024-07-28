DenseLo = $40
DenseHi = $41
Rows = $42
Cols = $43
SparseLo = $44
SparseHi = $45
CurrentCol = $46
CurrElementLo = $47
CurrElementHi = $48

* = $2800

!zone Matrix {

.OverflowError jam

; Input will always be 28 * 28
; Modifies X and Y
DenseToSparse
  jsr init
  ldx #0

.OuterLoop
  ldy #0

.InnerLoop
  ; Get current element index
  ; (cols * current row + current col)
  sty CurrentCol
  ldy Cols
  jsr umult8
  sta CurrElementHi
  clc
  lda CurrentCol
  adc prod_low
  sta CurrElementLo
  bcc NoCarry
  lda #1
  adc CurrElementHi
  sta CurrElementHi
  bvs .OverflowError
NoCarry
  ldy CurrentCol

  iny
  cpy $42
  bne .InnerLoop

  inx
  cpx $43
  bne .OuterLoop

  rts
}