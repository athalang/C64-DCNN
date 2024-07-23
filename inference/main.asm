*=$0801

!basic
  ldx #$00

ForwardsIter
  inx

  ; Forward

  cpx ModelBinary
  bne ForwardsIter
  rts

SampleImage
  !bin "../three.bin"

ModelBinary
  !bin "../model.bin"