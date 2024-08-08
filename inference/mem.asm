#importonce
#import "utils.asm"

* = $0901

memory: {

// Blocks 8 byte aligned
// Size - 2 bytes
// Least three bits used for options
// When free block,
// Previous free block - 2 bytes
// Next free block - 2 bytes
// Payload, then size + options repeated

.const HEAP_START	= $4000
.const HEAP_END		= $D000
.const HEAP_SIZE	= HEAP_END - HEAP_START
// Mask out unused bits and set free
.const HEAP_HEADER	= (HEAP_SIZE & %1111111111111000) | 1

@heap_init:	lda #%00110110 // Hide BASIC ROM
		sta 1

		str_immediate_u16_u16(HEAP_HEADER, HEAP_START)

		// Clear next & prev pointers
		str_immediate_u16_u16(0, HEAP_START+2)
		str_immediate_u16_u16(0, HEAP_START+4)

		str_immediate_u16_u16(HEAP_HEADER, HEAP_END-2)

		.byte JAM

} // End of scope memory