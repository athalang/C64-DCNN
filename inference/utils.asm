#importonce

.var JAM = $02

.macro add_u16 (n, m, res) {
        clc
        lda n
        adc m
        sta res
        lda n+1
        adc m+1
        sta res+1
}

* = $1000

model_params:
.import binary	"../model.bin"

dense_image:
.import binary	"../three.bin"