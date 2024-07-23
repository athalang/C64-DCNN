*=$0801
!basic

  ; Jam if 0 layers in model
  lda #0
  cmp ModelBinary
  beq FormatError

  ; Store 2 * num of layers at $02
  ldx #0
  lda ModelBinary
  asl
  bcs FormatError ; Fail if overflow
  sta $02

ForwardsIter
  ; Copy layer pointer to $03 and $04
  inx ; Pointers start from 1 relative to ModelBinary
  lda ModelBinary,x
  sta $03
  inx
  lda ModelBinary,x
  sta $04

  ; Forward

  cpx $02
  bne ForwardsIter

  rts

FormatError jam

SampleImage !bin "../three.bin"

ModelBinary !bin "../model.bin"