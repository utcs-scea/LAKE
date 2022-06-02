#include <stdint.h>
#include <unistd.h>

// ============ Definitions ============
/*!
 * XXH_CPU_LITTLE_ENDIAN:
 * Defined to 1 if the target is little endian, or 0 if it is big endian.
 * It can be defined externally, for example on the compiler command line.
 *
 * If it is not defined, a runtime check (which is usually constant folded)
 * is used instead.
 */
#ifndef XXH_CPU_LITTLE_ENDIAN
/*
 * Try to detect endianness automatically, to avoid the nonstandard behavior
 * in `XXH_isLittleEndian()`
 */
#  if defined(_WIN32) /* Windows is always little endian */ \
     || defined(__LITTLE_ENDIAN__) \
     || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#    define XXH_CPU_LITTLE_ENDIAN 1
#  elif defined(__BIG_ENDIAN__) \
     || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#    define XXH_CPU_LITTLE_ENDIAN 0
#  else
static int XXH_isLittleEndian(void)
{
    /*
     * Nonstandard, but well-defined behavior in practice.
     * Don't use static: it is detrimental to performance.
     */
    const union { xxh_u32 u; xxh_u8 c[4]; } one = { 1 };
    return one.c[0];
}

#   define XXH_CPU_LITTLE_ENDIAN   XXH_isLittleEndian()
#  endif
#endif
#define XXH_rotl32(x,r) (((x) << (r)) | ((x) >> (32 - (r))))
// TODO: does little-endian stuff affect us?
#define XXH_get32bits(p) XXH_readLE32_align(p, align)
#define XXH_ASSERT(c)   assert(c)
typedef uint8_t  xxh_u8;
typedef uint32_t XXH32_hash_t;
typedef XXH32_hash_t xxh_u32;
typedef enum { XXH_aligned, XXH_unaligned } XXH_alignment;
#ifndef XXH_REROLL
#  if defined(__OPTIMIZE_SIZE__)
#    define XXH_REROLL 1
#  else
#    define XXH_REROLL 0
#  endif
#endif
#if defined(_MSC_VER)     /* Visual Studio */
#  define XXH_swap32 _byteswap_ulong
#elif XXH_GCC_VERSION >= 403
#  define XXH_swap32 __builtin_bswap32
#else
static xxh_u32 XXH_swap32 (xxh_u32 x)
{
    return  ((x << 24) & 0xff000000 ) |
            ((x <<  8) & 0x00ff0000 ) |
            ((x >>  8) & 0x0000ff00 ) |
            ((x >> 24) & 0x000000ff );
}
#endif

__device__ xxh_u32 XXH_readLE32_align(const void* ptr, XXH_alignment align)
{
  return XXH_CPU_LITTLE_ENDIAN ? *(const xxh_u32*)ptr : XXH_swap32(*(const xxh_u32*)ptr);
}


static const xxh_u32 PRIME32_1 = 0x9E3779B1U;   /* 0b10011110001101110111100110110001 */
static const xxh_u32 PRIME32_2 = 0x85EBCA77U;   /* 0b10000101111010111100101001110111 */
static const xxh_u32 PRIME32_3 = 0xC2B2AE3DU;   /* 0b11000010101100101010111000111101 */
static const xxh_u32 PRIME32_4 = 0x27D4EB2FU;   /* 0b00100111110101001110101100101111 */
static const xxh_u32 PRIME32_5 = 0x165667B1U;   /* 0b00010110010101100110011110110001 */

// ============ End definitions ============

/* mix all bits */
__device__ xxh_u32 XXH32_avalanche(xxh_u32 h32)
{
    h32 ^= h32 >> 15;
    h32 *= PRIME32_2;
    h32 ^= h32 >> 13;
    h32 *= PRIME32_3;
    h32 ^= h32 >> 16;
    return(h32);
}

__device__ xxh_u32 XXH32_finalize(xxh_u32 h32, const xxh_u8* ptr, size_t len, XXH_alignment align)
{
#define PROCESS1               \
    h32 += (*ptr++) * PRIME32_5; \
    h32 = XXH_rotl32(h32, 11) * PRIME32_1 ;

#define PROCESS4                         \
    h32 += XXH_get32bits(ptr) * PRIME32_3; \
    ptr+=4;                                \
    h32  = XXH_rotl32(h32, 17) * PRIME32_4 ;

    /* Compact rerolled version */
    if (XXH_REROLL) {
        len &= 15;
        while (len >= 4) {
            PROCESS4;
            len -= 4;
        }
        while (len > 0) {
            PROCESS1;
            --len;
        }
        return XXH32_avalanche(h32);
    } else {
         switch(len&15) /* or switch(bEnd - p) */ {
           case 12:      PROCESS4;
                         /* fallthrough */
           case 8:       PROCESS4;
                         /* fallthrough */
           case 4:       PROCESS4;
                         return XXH32_avalanche(h32);

           case 13:      PROCESS4;
                         /* fallthrough */
           case 9:       PROCESS4;
                         /* fallthrough */
           case 5:       PROCESS4;
                         PROCESS1;
                         return XXH32_avalanche(h32);

           case 14:      PROCESS4;
                         /* fallthrough */
           case 10:      PROCESS4;
                         /* fallthrough */
           case 6:       PROCESS4;
                         PROCESS1;
                         PROCESS1;
                         return XXH32_avalanche(h32);

           case 15:      PROCESS4;
                         /* fallthrough */
           case 11:      PROCESS4;
                         /* fallthrough */
           case 7:       PROCESS4;
                         /* fallthrough */
           case 3:       PROCESS1;
                         /* fallthrough */
           case 2:       PROCESS1;
                         /* fallthrough */
           case 1:       PROCESS1;
                         /* fallthrough */
           case 0:       return XXH32_avalanche(h32);
        }
        XXH_ASSERT(0);
        return h32;   /* reaching this point is deemed impossible */
    }
}

__device__ xxh_u32 XXH32_round(xxh_u32 acc, xxh_u32 input)
{
    acc += input * PRIME32_2;
    acc  = XXH_rotl32(acc, 13);
    acc *= PRIME32_1;
// TODO: look at this in more detail
#if 0 && defined(__GNUC__) && defined(__SSE4_1__) && !defined(XXH_ENABLE_AUTOVECTORIZE)
    /*
     * UGLY HACK:
     * This inline assembly hack forces acc into a normal register. This is the
     * only thing that prevents GCC and Clang from autovectorizing the XXH32
     * loop (pragmas and attributes don't work for some resason) without globally
     * disabling SSE4.1.
     *
     * The reason we want to avoid vectorization is because despite working on
     * 4 integers at a time, there are multiple factors slowing XXH32 down on
     * SSE4:
     * - There's a ridiculous amount of lag from pmulld (10 cycles of latency on
     *   newer chips!) making it slightly slower to multiply four integers at
     *   once compared to four integers independently. Even when pmulld was
     *   fastest, Sandy/Ivy Bridge, it is still not worth it to go into SSE
     *   just to multiply unless doing a long operation.
     *
     * - Four instructions are required to rotate,
     *      movqda tmp,  v // not required with VEX encoding
     *      pslld  tmp, 13 // tmp <<= 13
     *      psrld  v,   19 // x >>= 19
     *      por    v,  tmp // x |= tmp
     *   compared to one for scalar:
     *      roll   v, 13    // reliably fast across the board
     *      shldl  v, v, 13 // Sandy Bridge and later prefer this for some reason
     *
     * - Instruction level parallelism is actually more beneficial here because
     *   the SIMD actually serializes this operation: While v1 is rotating, v2
     *   can load data, while v3 can multiply. SSE forces them to operate
     *   together.
     *
     * How this hack works:
     * __asm__(""       // Declare an assembly block but don't declare any instructions
     *          :       // However, as an Input/Output Operand,
     *          "+r"    // constrain a read/write operand (+) as a general purpose register (r).
     *          (acc)   // and set acc as the operand
     * );
     *
     * Because of the 'r', the compiler has promised that seed will be in a
     * general purpose register and the '+' says that it will be 'read/write',
     * so it has to assume it has changed. It is like volatile without all the
     * loads and stores.
     *
     * Since the argument has to be in a normal register (not an SSE register),
     * each time XXH32_round is called, it is impossible to vectorize.
     */
    __asm__("" : "+r" (acc));
#endif
    return acc;
}

__device__ xxh_u32 XXH32_endian_align(const xxh_u8* input, size_t len, xxh_u32 seed, XXH_alignment align)
{
    const xxh_u8* bEnd = input + len;
    xxh_u32 h32;

#if defined(XXH_ACCEPT_NULL_INPUT_POINTER) && (XXH_ACCEPT_NULL_INPUT_POINTER>=1)
    if (input==NULL) {
        len=0;
        bEnd=input=(const xxh_u8*)(size_t)16;
    }
#endif

    if (len>=16) {
        const xxh_u8* const limit = bEnd - 15;
        xxh_u32 v1 = seed + PRIME32_1 + PRIME32_2;
        xxh_u32 v2 = seed + PRIME32_2;
        xxh_u32 v3 = seed + 0;
        xxh_u32 v4 = seed - PRIME32_1;

        do {
            v1 = XXH32_round(v1, XXH_get32bits(input)); input += 4;
            v2 = XXH32_round(v2, XXH_get32bits(input)); input += 4;
            v3 = XXH32_round(v3, XXH_get32bits(input)); input += 4;
            v4 = XXH32_round(v4, XXH_get32bits(input)); input += 4;
        } while (input < limit);

        h32 = XXH_rotl32(v1, 1)  + XXH_rotl32(v2, 7)
            + XXH_rotl32(v3, 12) + XXH_rotl32(v4, 18);
    } else {
        h32  = seed + PRIME32_5;
    }

    h32 += (xxh_u32)len;

    return XXH32_finalize(h32, input, len&15, align);
}

// TODO: for now, make len and seed deterministic
__global__ void XXH32(void *input, XXH32_hash_t *output)
{
  size_t len = 4096; 
  XXH32_hash_t seed = 17;
  int idx =
    blockIdx.x * blockDim.x + threadIdx.x +
    (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + 
    (blockIdx.z * blockDim.z + threadIdx.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y;

  // Calculate the offset of the page to be hashed
  char *in_addr = ((char *) input) + idx * len;
  uint32_t *out_addr = ((uint32_t *) output) + idx;
#if 0
    /* Simple version, good for code maintenance, but unfortunately slow for small inputs */
    XXH32_state_t state;
    XXH32_reset(&state, seed);
    XXH32_update(&state, (const xxh_u8*)input, len);
    return XXH32_digest(&state);

#else
  *out_addr = XXH32_endian_align((const xxh_u8*) in_addr, len, seed, XXH_aligned);
#endif
}


// workspace must be 4 * (# of pages) * 4 bytes
// xxh_u32 v1 = 17 + PRIME32_1 + PRIME32_2;
// xxh_u32 v2 = 17 + PRIME32_2;
// xxh_u32 v3 = 17 + 0;
// xxh_u32 v4 = 17 - PRIME32_1;

__global__ void XXH32v2(void *input, XXH32_hash_t *output, uint32_t* workspace, uint32_t seed, uint32_t* seeds)
{
    size_t page_size = 4096; 
    //XXH32_hash_t seed = 17;
    int idx =
        blockIdx.x * blockDim.x + threadIdx.x +
        (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + 
        (blockIdx.z * blockDim.z + threadIdx.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y;

    int page_offset = idx / 4;
    int word_offset = idx % 4;
    int ws_offset = (page_offset * 4) + word_offset;

    XXH_alignment align = XXH_aligned;

    // Calculate the offset of the page to be hashed
    char *page_addr = ((char *) input) + (page_offset * page_size);

    const xxh_u8* page_end = (const xxh_u8*) page_addr + page_size;
    const xxh_u8* const limit = page_end - 15;
    xxh_u32 v = seeds[word_offset]; 

    xxh_u8* thread_input = (xxh_u8*) (page_addr + (word_offset*4));
    do {
        v = XXH32_round(v, XXH_get32bits(thread_input)); 
        thread_input += 4;
    } while (thread_input < limit);
    workspace[ws_offset] = v;

    __syncthreads();

    if (word_offset == 0) {
        v = XXH_rotl32(workspace[ws_offset], 1)  + XXH_rotl32(workspace[ws_offset+1], 7)
            + XXH_rotl32(workspace[ws_offset+2], 12) + XXH_rotl32(workspace[ws_offset+3], 18);
        v += (xxh_u32)page_size;
        //finalize doesnt do anything on 4k pages
        //v = XXH32_finalize(h32, input, page_size&15, align);
        v = XXH32_avalanche(v);
        uint32_t *out_addr = ((uint32_t *) output) + page_offset;
        *out_addr = XXH32_endian_align((const xxh_u8*) page_addr, page_size, seed, XXH_aligned);
    }        
}

