ForwardIterMax = $08
CurrentLayerLo = $09
CurrentLayerHi = $0A

!ifndef libMatrixImported {
libMatrixImported
  !src "inference/matrix.asm"
}
!ifndef libMultImported {
libMultImported
  !src "inference/multiply.asm"
}

* = $0801

SampleImage !bin "../three.bin"
ModelBinary !bin "../model.bin"

!zone Main {

.FormatError jam
.OverflowError jam

  ; Switch out BASIC ROM
  lda #%00110110
  sta $01

  ; Set up umult16 routine
  jsr init

  ; Jam if 0 layers in model
  lda #0
  cmp ModelBinary
  beq .FormatError

  ; Store 2 * num of layers at ForwardIterMax
  ldx #0
  lda ModelBinary
  asl
  bcs .OverflowError ; Jam if overflow
  sta ForwardIterMax

ForwardIter
  ; Copy layer pointer to $03 and $04
  inx ; Pointers start from 1 relative to ModelBinary
  lda ModelBinary,x
  sta CurrentLayerLo
  inx
  lda ModelBinary,x
  sta CurrentLayerHi

  ; Forward

  cpx ForwardIterMax
  bne ForwardIter

  rts
}