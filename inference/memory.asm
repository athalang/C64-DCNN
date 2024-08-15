#importonce
#import "utils.asm"

// Blocks 8 byte aligned
// Size + 2 unused bits + Free boolean = 2 bytes
// If free block, prev and next pointers
// Payload, then size + options repeated (Knuth boundary tags)

.const BITS_IGNORED	= 3
.const BYTE_ALIGNMENT	= pow(2, BITS_IGNORED)
// Get the u8 mask for the least BITS_IGNORED bits
.const IGNORED_MASK	= BYTE_ALIGNMENT - 1

.const BLOCKS		= $1000

.const HEAP_START	= $5000
.const HEAP_SIZE	= BLOCKS * BYTE_ALIGNMENT

.assert "Heap start aligned?", mod(HEAP_START, BYTE_ALIGNMENT) == 0, true

// Plus 1 after the end of the heap, i.e. first byte not in heap
.const HEAP_END_1P	= HEAP_START + HEAP_SIZE

// Remove unused bits and set free
.const HEAP_OPTIONS	= (HEAP_SIZE >> BITS_IGNORED << BITS_IGNORED) | 1

.const FIRST_FREE	= $40	// 2 bytes
.const MALLOC_SIZE	= $42	// 2 bytes
.const CURR_BLOCK	= $44	// 2 bytes
.const CURR_BLOCK_SIZE	= $46	// 2 bytes
.const TEMP		= $48	// 2 bytes

* = $0901

memory: {

@heap_init:	lda #%00110110 // Hide BASIC ROM
		sta 1

		// Set first free pointer
		str_immediate_u16_u16(HEAP_START, FIRST_FREE)

		// Create a free block that spans the entire heap
		str_immediate_u16_u16(HEAP_OPTIONS, HEAP_START)

		// Clear next/prev pointers
		str_immediate_u16_u16(0, HEAP_START+2)
		str_immediate_u16_u16(0, HEAP_START+4)

		str_immediate_u16_u16(HEAP_OPTIONS, HEAP_END_1P-2)

		// Debug malloc
		str_immediate_u16_u16(9, MALLOC_SIZE)
		jsr malloc

		.byte JAM

		rts

// Modifies Y
@malloc:	// Align if unaligned
!:		lda #IGNORED_MASK
		and MALLOC_SIZE
		beq !+
		add_immediate_u16(MALLOC_SIZE, 1, 0, MALLOC_SIZE)
		jmp !-

!:		str_absolute_u16_u16(FIRST_FREE, CURR_BLOCK)

!loop:		ldy #0
		lda (CURR_BLOCK),y

		// Remove unused bits
		.for(var i = 0; i < BITS_IGNORED; i++) {
			lsr
		}
		.for(var i = 0; i < BITS_IGNORED; i++) {
			asl
		}

		sta CURR_BLOCK_SIZE
		ldy #1
		lda (CURR_BLOCK),y
		sta CURR_BLOCK_SIZE+1

		cmp_absolute_u16(CURR_BLOCK_SIZE, MALLOC_SIZE, TEMP)
		// When malloc size >= current block size
		bpl block_found

		// Set current pointer to next
		ldy #4
		lda (CURR_BLOCK),y
		sta TEMP
		ldy #5
		lda (CURR_BLOCK),y
		sta TEMP+1
		str_absolute_u16_u16(TEMP, CURR_BLOCK)

		cmp_immediate_u16(0, CURR_BLOCK, TEMP)
		bne !loop- // If not null ptr, continue
		.byte JAM

block_found:	rts

} // End of scope memory

* = $4000