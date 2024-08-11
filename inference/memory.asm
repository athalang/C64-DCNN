#importonce
#import "utils.asm"

// Blocks 8 byte aligned
// Size + 2 unused bits + Free boolean = 2 bytes
// Payload, then size + options repeated (Knuth boundary tags)

.const BITS_IGNORED	= 3
.const BLOCK_ALIGNMENT	= pow(2, BITS_IGNORED)
.const BLOCKS		= $1000

.const HEAP_START	= $5000
.const HEAP_SIZE	= BLOCKS * BLOCK_ALIGNMENT

.assert "Heap start aligned?", mod(HEAP_START, BLOCK_ALIGNMENT) == 0, true

// Plus 1 after the end of the heap, i.e. first byte not in heap
.const HEAP_END_1P	= HEAP_START + HEAP_SIZE

// Remove unused bits and set free
.const HEAP_OPTIONS	= (HEAP_SIZE >> BITS_IGNORED << BITS_IGNORED) | 1

* = $0901

memory: {

@heap_init:	lda #%00110110 // Hide BASIC ROM
		sta 1

		str_immediate_u16_u16(HEAP_OPTIONS, HEAP_START)

		// Clear next/prev pointers
		str_immediate_u16_u16(0, HEAP_START+2)
		str_immediate_u16_u16(0, HEAP_START+4)

		str_immediate_u16_u16(HEAP_OPTIONS, HEAP_END_1P-2)

		.byte JAM

		rts

} // End of scope memory

* = $4000