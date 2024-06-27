*=$0801

!basic
  ldx #$00

LOOP
  lda TEXT,x
  jsr $FFD2
  inx
  cmp #$00
  bne LOOP
  rts

TEXT
  !text "{clr}HELLO WORLD!",0