#importonce
#import "utils.asm"

// Blocks 8 byte aligned
// Size + 2 unused bits + Free boolean = 2 bytes
// Payload, then size + options repeated

.const BLOCK_ALIGNMENT	= 8
.const BLOCKS		= $1000		// Must be divisble by 2

// Free list
.const FREE_START	= $4000		// Needs to be aligned
// Worst case, when every other block is free
.const FREE_WORST_CASE	= BLOCKS / 2
// How much is permissible, i.e. reasonable rank limit
.const FREE_MAX		= FREE_WORST_CASE / 4

.const HEAP_START	= FREE_START + FREE_MAX
.const HEAP_SIZE	= BLOCKS * BLOCK_ALIGNMENT
// Plus 1 after the end of the heap, i.e. first byte not in heap
.const HEAP_END_1P	= HEAP_START + HEAP_SIZE

// Mask out unused bits and set free
.const HEAP_OPTIONS	= (HEAP_SIZE & %1111111111111000) | 1

* = $0901

memory: {

@heap_init:	lda #%00110110 // Hide BASIC ROM
		sta 1

		str_immediate_u16_u16(HEAP_OPTIONS, HEAP_START)
		str_immediate_u16_u16(HEAP_OPTIONS, HEAP_END_1P-2)

		.byte JAM

		rts

} // End of scope memory

* = $4000