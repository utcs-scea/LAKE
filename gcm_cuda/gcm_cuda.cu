#include "aes_gcm.h"
#include <openssl/rand.h>
#include <openssl/evp.h>
#include <assert.h>

/*****************************************************************************/
/* Defines:                                                                  */
/*****************************************************************************/
// The number of columns comprising a state in AES. This is a constant in AES. Value=4
#define Nb 4
#define Nk 8
#define Nr 14

/*****************************************************************************/
/* Private variables:                                                        */
/*****************************************************************************/
// state - array holding the intermediate results during decryption.
typedef uint8_t state_t[4][4];

// The lookup-tables are marked const so they can be placed in read-only storage instead of RAM
// The numbers below can be computed dynamically trading ROM for RAM -
// This can be useful in (embedded) bootloader applications, where ROM is often limited.
static const uint8_t sbox_host[256] = {
  //0     1    2      3     4    5     6     7      8    9     A      B    C     D     E     F
  0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
  0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
  0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
  0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
  0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
  0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
  0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
  0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
  0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
  0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
  0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
  0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
  0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
  0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
  0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
  0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16 };

static const uint8_t rsbox_host[256] = {
  0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
  0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
  0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
  0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
  0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
  0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
  0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
  0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
  0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
  0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
  0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
  0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
  0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
  0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
  0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
  0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d };

// The round constant word array, Rcon[i], contains the values given by
// x to the power (i-1) being powers of x (x is denoted as {02}) in the field GF(2^8)
static const uint8_t Rcon_host[11] = {
  0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36 };

/*
 * Jordan Goulder points out in PR #12 (https://github.com/kokke/tiny-AES-C/pull/12),
 * that you can remove most of the elements in the Rcon array, because they are unused.
 *
 * From Wikipedia's article on the Rijndael key schedule @ https://en.wikipedia.org/wiki/Rijndael_key_schedule#Rcon
 *
 * "Only the first some of these constants are actually used – up to rcon[10] for AES-128 (as 11 round keys are needed),
 *  up to rcon[8] for AES-192, up to rcon[7] for AES-256. rcon[0] is not used in AES algorithm."
 */


/*****************************************************************************/
/* Private functions:                                                        */
/*****************************************************************************/

// This function produces Nb(Nr+1) round keys. The round keys are used in each round to decrypt the states.
__device__ void KeyExpansion(const uint8_t* sbox, const uint8_t* Rcon, uint8_t* RoundKey,
    const uint8_t* Key)
{
  unsigned i, j, k;
  uint8_t tempa[4]; // Used for the column/row operations

  // The first round key is the key itself.
  for (i = 0; i < Nk; ++i)
  {
    RoundKey[(i * 4) + 0] = Key[(i * 4) + 0];
    RoundKey[(i * 4) + 1] = Key[(i * 4) + 1];
    RoundKey[(i * 4) + 2] = Key[(i * 4) + 2];
    RoundKey[(i * 4) + 3] = Key[(i * 4) + 3];
  }

  // All other round keys are found from the previous round keys.
  for (i = Nk; i < Nb * (Nr + 1); ++i)
  {
    {
      k = (i - 1) * 4;
      tempa[0]=RoundKey[k + 0];
      tempa[1]=RoundKey[k + 1];
      tempa[2]=RoundKey[k + 2];
      tempa[3]=RoundKey[k + 3];

    }

    if (i % Nk == 0)
    {
      // This function shifts the 4 bytes in a word to the left once.
      // [a0,a1,a2,a3] becomes [a1,a2,a3,a0]

      // Function RotWord()
      {
        const uint8_t u8tmp = tempa[0];
        tempa[0] = tempa[1];
        tempa[1] = tempa[2];
        tempa[2] = tempa[3];
        tempa[3] = u8tmp;
      }

      // SubWord() is a function that takes a four-byte input word and
      // applies the S-box to each of the four bytes to produce an output word.

      // Function Subword()
      {
        tempa[0] = sbox[tempa[0]];
        tempa[1] = sbox[tempa[1]];
        tempa[2] = sbox[tempa[2]];
        tempa[3] = sbox[tempa[3]];
      }

      tempa[0] = tempa[0] ^ Rcon[i/Nk];
    }
    if (i % Nk == 4)
    {
      // Function Subword()
      {
        tempa[0] = sbox[tempa[0]];
        tempa[1] = sbox[tempa[1]];
        tempa[2] = sbox[tempa[2]];
        tempa[3] = sbox[tempa[3]];
      }
    }
    j = i * 4; k=(i - Nk) * 4;
    RoundKey[j + 0] = RoundKey[k + 0] ^ tempa[0];
    RoundKey[j + 1] = RoundKey[k + 1] ^ tempa[1];
    RoundKey[j + 2] = RoundKey[k + 2] ^ tempa[2];
    RoundKey[j + 3] = RoundKey[k + 3] ^ tempa[3];
  }
}

// This function adds the round key to state.
// The round key is added to the state by an XOR function.
__device__ void AddRoundKey(uint8_t r, state_t* state, const uint8_t* RoundKey)
{
  uint8_t i,j;
#pragma unroll
  for (i = 0; i < 4; ++i)
  {
#pragma unroll
    for (j = 0; j < 4; ++j)
    {
      (*state)[i][j] ^= RoundKey[(r * Nb * 4) + (i * Nb) + j];
    }
  }
}

// The SubBytes Function Substitutes the values in the
// state matrix with values in an S-box.
__device__ void SubBytes(const uint8_t* sbox, state_t* state)
{
  uint8_t i, j;
#pragma unroll
  for (i = 0; i < 4; ++i)
  {
#pragma unroll
    for (j = 0; j < 4; ++j)
    {
      (*state)[j][i] = sbox[(*state)[j][i]];
    }
  }
}

// The ShiftRows() function shifts the rows in the state to the left.
// Each row is shifted with different offset.
// Offset = Row number. So the first row is not shifted.
__device__ void ShiftRows(state_t* state)
{
  uint8_t temp;

  // Rotate first row 1 columns to left
  temp           = (*state)[0][1];
  (*state)[0][1] = (*state)[1][1];
  (*state)[1][1] = (*state)[2][1];
  (*state)[2][1] = (*state)[3][1];
  (*state)[3][1] = temp;

  // Rotate second row 2 columns to left
  temp           = (*state)[0][2];
  (*state)[0][2] = (*state)[2][2];
  (*state)[2][2] = temp;

  temp           = (*state)[1][2];
  (*state)[1][2] = (*state)[3][2];
  (*state)[3][2] = temp;

  // Rotate third row 3 columns to left
  temp           = (*state)[0][3];
  (*state)[0][3] = (*state)[3][3];
  (*state)[3][3] = (*state)[2][3];
  (*state)[2][3] = (*state)[1][3];
  (*state)[1][3] = temp;
}

__device__ uint8_t xtime(uint8_t x)
{
  return ((x<<1) ^ (((x>>7) & 1) * 0x1b));
}

// MixColumns function mixes the columns of the state matrix
__device__ void MixColumns(state_t* state)
{
  uint8_t i;
  uint8_t Tmp, Tm, t;
#pragma unroll
  for (i = 0; i < 4; ++i)
  {
    t   = (*state)[i][0];
    Tmp = (*state)[i][0] ^ (*state)[i][1] ^ (*state)[i][2] ^ (*state)[i][3] ;
    Tm  = (*state)[i][0] ^ (*state)[i][1] ; Tm = xtime(Tm);  (*state)[i][0] ^= Tm ^ Tmp ;
    Tm  = (*state)[i][1] ^ (*state)[i][2] ; Tm = xtime(Tm);  (*state)[i][1] ^= Tm ^ Tmp ;
    Tm  = (*state)[i][2] ^ (*state)[i][3] ; Tm = xtime(Tm);  (*state)[i][2] ^= Tm ^ Tmp ;
    Tm  = (*state)[i][3] ^ t ;              Tm = xtime(Tm);  (*state)[i][3] ^= Tm ^ Tmp ;
  }
}


// Cipher is the main function that encrypts the PlainText.
__device__ void Cipher(const uint8_t* sbox, state_t* state, const uint8_t* RoundKey)
{
  uint8_t r = 0;
  state_t local_state;
#pragma unroll
  for (int i = 0; i < 4; i++) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      local_state[i][j] = (*state)[i][j];
    }
  }

  // Add the First round key to the state before starting the rounds.
  AddRoundKey(0, &local_state, RoundKey);

  // There will be Nr rounds.
  // The first Nr-1 rounds are identical.
  // These Nr-1 rounds are executed in the loop below.
#pragma unroll
  for (r = 1; r < Nr; ++r)
  {
    SubBytes(sbox, &local_state);
    ShiftRows(&local_state);
    MixColumns(&local_state);
    AddRoundKey(r, &local_state, RoundKey);
  }

  // The last round is given below.
  // The MixColumns function is not here in the last round.
  SubBytes(sbox, &local_state);
  ShiftRows(&local_state);
  AddRoundKey(Nr, &local_state, RoundKey);

#pragma unroll
  for (int i = 0; i < 4; i++) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      (*state)[i][j] = local_state[i][j];
    }
  }
}

__global__ void AES_key_expansion_kernel(const uint8_t* sbox, const uint8_t* Rcon,
    const uint8_t* key, uint8_t* roundkey) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    KeyExpansion(sbox, Rcon, roundkey, key);
  }
}

__global__ void AES_encrypt_one_block_kernel(const uint8_t* sbox, const uint8_t* roundkey,
    uint8_t* data) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    Cipher(sbox, (state_t*)data, roundkey);
  }
}

static const uint64_t gf_last4_host[16] = {
  0x0000, 0x1c20, 0x3840, 0x2460, 0x7080, 0x6ca0, 0x48c0, 0x54e0,
  0xe100, 0xfd20, 0xd940, 0xc560, 0x9180, 0x8da0, 0xa9c0, 0xb5e0  };

__device__ uint32_t get_be32(const uint8_t *a)
{
  return ((uint32_t) a[0] << 24) | ((uint32_t) a[1] << 16) | ((uint32_t) a[2] << 8) | a[3];
}

__device__ void put_be32(uint8_t *a, uint32_t val)
{
  a[0] = (val >> 24) & 0xff;
  a[1] = (val >> 16) & 0xff;
  a[2] = (val >> 8) & 0xff;
  a[3] = val & 0xff;
}

__device__ void gf_mult_fast(const uint64_t* last4, const uint64_t* HL, const uint64_t* HH,
    const uint8_t* x, uint8_t* output) {
  int i;
  uint8_t lo, hi, rem;
  uint64_t zh, zl;
  lo = (uint8_t)( x[15] & 0x0f );
  hi = (uint8_t)( x[15] >> 4 );
  zh = HH[lo];
  zl = HL[lo];
#pragma unroll
  for( i = 15; i >= 0; i-- ) {
    lo = (uint8_t) ( x[i] & 0x0f );
    hi = (uint8_t) ( x[i] >> 4 );
    if( i != 15 ) {
      rem = (uint8_t) ( zl & 0x0f );
      zl = ( zh << 60 ) | ( zl >> 4 );
      zh = ( zh >> 4 );
      zh ^= (uint64_t) last4[rem] << 48;
      zh ^= HH[lo];
      zl ^= HL[lo];
    }
    rem = (uint8_t) ( zl & 0x0f );
    zl = ( zh << 60 ) | ( zl >> 4 );
    zh = ( zh >> 4 );
    zh ^= (uint64_t) last4[rem] << 48;
    zh ^= HH[hi];
    zl ^= HL[hi];
  }
  put_be32(output, zh >> 32);
  put_be32(output + 4, zh);
  put_be32(output + 8, zl >> 32);
  put_be32(output + 12, zl);
}

__device__ void gf_build_table(const uint8_t* h, uint64_t* HL, uint64_t* HH) {
  int i, j;
  uint64_t hi, lo;
  uint64_t vl, vh;

  hi = get_be32(h);
  lo = get_be32(h + 4);
  vh = (uint64_t) hi << 32 | lo;

  hi = get_be32(h + 8);
  lo = get_be32(h + 12);
  vl = (uint64_t) hi << 32 | lo;

  HL[8] = vl;                // 8 = 1000 corresponds to 1 in GF(2^128)
  HH[8] = vh;
  HH[0] = 0;                 // 0 corresponds to 0 in GF(2^128)
  HL[0] = 0;

  for( i = 4; i > 0; i >>= 1 ) {
    uint32_t T = (uint32_t) ( vl & 1 ) * 0xe1000000U;
    vl  = ( vh << 63 ) | ( vl >> 1 );
    vh  = ( vh >> 1 ) ^ ( (uint64_t) T << 32);
    HL[i] = vl;
    HH[i] = vh;
  }
  for (i = 2; i < 16; i <<= 1 ) {
    uint64_t *HiL = HL + i, *HiH = HH + i;
    vh = *HiH;
    vl = *HiL;
    for( j = 1; j < i; j++ ) {
      HiH[j] = vh ^ HH[j];
      HiL[j] = vl ^ HL[j];
    }
  }
}

__global__ void AES_GCM_setup_gf_mult_table_kernel(
    const uint64_t* last4, const uint8_t* h, uint64_t* HL, uint64_t* HH,
    uint64_t* HL_long, uint64_t* HH_long,
    uint64_t* HL_sqr_long, uint64_t* HH_sqr_long) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    gf_build_table(h, HL, HH);
    uint8_t h_long[16];
    uint8_t tmp[16];
    for (int i = 0; i < 16; i++) {
      h_long[i] = h[i];
    }
    for (int i = 0; i < AES_GCM_STEP - 1; i++) {
      gf_mult_fast(last4, HL, HH, h_long, tmp);
      for (int j = 0; j < 16; j++) {
        h_long[j] = tmp[j];
      }
    }
    gf_build_table(h_long, HL_long, HH_long);
    uint8_t h_sqr_long[16];
    for (int i = 0; i < 16; i++) {
      h_sqr_long[i] = h_long[i];
    }
    for (int i = 0; i < AES_GCM_STEP - 1; i++) {
      gf_mult_fast(last4, HL_long, HH_long, h_sqr_long, tmp);
      for (int j = 0; j < 16; j++) {
        h_sqr_long[j] = tmp[j];
      }
    }
    gf_build_table(h_sqr_long, HL_sqr_long, HH_sqr_long);
  }
}

__global__ void AES_GCM_xcrypt_kernel(uint8_t* dst, const uint8_t* sbox, const uint8_t* roundkey,
    const uint8_t* nonce, const uint8_t* src, uint32_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid * AES_BLOCKLEN < size) {
    uint8_t buffer[16];
#pragma unroll
    for (int i = 0; i < 12; i++) {
      buffer[i] = nonce[i];
    }
    put_be32(buffer + 12, (uint32_t)tid + 2);
    Cipher(sbox, (state_t*)buffer, roundkey);
#pragma unroll
    for (int i = 0; i < AES_BLOCKLEN; i++) {
      dst[tid*AES_BLOCKLEN+i] = src[tid*AES_BLOCKLEN+i] ^ buffer[i];
    }
  }
}

__global__ void AES_GCM_mac_kernel(
    uint64_t* last4, uint64_t* HL, uint64_t* HH, int num_parts,
    uint8_t* input, uint32_t num_block, uint8_t* output) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_parts) {
    return;
  }
  __shared__ uint64_t local_last4[16];
  __shared__ uint64_t local_HL[16];
  __shared__ uint64_t local_HH[16];
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < 16; j++) {
      local_last4[j] = last4[j];
      local_HL[j] = HL[j];
      local_HH[j] = HH[j];
    }
  }
  __syncthreads();
  uint8_t v[16], u[16];
#pragma unroll
  for (int j = 0; j < 16; j++) {
    v[j] = 0;
  }
  int head_size = num_block % num_parts;
  if (tid >= num_parts - head_size) {
    int block_id = head_size + tid - num_parts;
#pragma unroll
    for (int j = 0; j < 16; j++) {
      v[j] = input[block_id * 16 + j];
    }
  }
  int i = head_size + tid;
#pragma unroll 16
  while (i < num_block) {
    gf_mult_fast(local_last4, local_HL, local_HH, v, u);
#pragma unroll
    for (int j = 0; j < 16; j++) {
      v[j] = u[j] ^ input[i * 16 + j];
    }
    i += num_parts;
  }
#pragma unroll
  for (int j = 0; j < 16; j++) {
    output[tid * 16 + j] = v[j];
  }
}

__global__ void AES_GCM_mac_final_kernel(
    const uint64_t* last4, uint64_t* HL, uint64_t* HH, uint8_t* sbox, uint8_t* roundkey,
    uint8_t* nonce, uint8_t* x, uint32_t input_size, uint8_t* mac) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    uint8_t u[16], v[16];
    gf_mult_fast(last4, HL, HH, x, v);
#pragma unroll
    for (int i = 0; i < 16; i++) {
      u[i] = 0;
    }
    put_be32(u + 12, input_size * 8);
#pragma unroll
    for (int i = 0; i < 16; i++) {
      u[i] ^= v[i];
    }
    gf_mult_fast(last4, HL, HH, u, v);
    for (int i = 0; i < 12; i++) {
      u[i] = nonce[i];
    }
    put_be32(u + 12, (uint32_t)1);
    Cipher(sbox, (state_t*)u, roundkey);
#pragma unroll
    for (int i = 0; i < 16; i++) {
      mac[i] = v[i] ^ u[i];
    }
  }
}

__global__ void AES_GCM_next_nonce_kernel(uint8_t* nonce) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    int i = 0;
    while (i < 12) {
      nonce[i]++;
      if (nonce[i] > 0) break;
      i++;
    }
  }
}

void AES_GCM_xcrypt(uint8_t* dst, const uint8_t* nonce,
    const uint8_t* src, uint32_t size) {
  int num_block = (size / 16 + kBaseThreadNum-1) / kBaseThreadNum;
//   hipLaunchAddToBatch(batch, HIP_KERNEL_NAME(AES_GCM_xcrypt_kernel), num_block, kBaseThreadNum, 0, stream,
//       dst, engine->sbox, engine->aes_roundkey, nonce, src, size);
  dim3 numBlocks(num_block);
  dim3 dimBlocks(kBaseThreadNum);
  AES_GCM_xcrypt_kernel<<<numBlocks, dimBlocks>>>(dst, d_engine->sbox, d_engine->aes_roundkey, nonce,
     src, size);
}

void AES_GCM_encrypt_one_block(uint8_t* data) {  
//   hipLaunchNOW(HIP_KERNEL_NAME(AES_encrypt_one_block_kernel), 1, 1, 0, stream,
//       engine->sbox, engine->aes_roundkey, data);
    dim3 numBlocks(1);
    dim3 dimBlocks(1);
    AES_encrypt_one_block_kernel<<<numBlocks, dimBlocks>>>(d_engine->sbox, d_engine->aes_roundkey, data);
}

void AES_GCM_compute_mac( uint8_t* dst, uint8_t* nonce,
    uint8_t* src, uint32_t size) {
//   hipLaunchAddToBatch(batch, HIP_KERNEL_NAME(AES_GCM_mac_kernel), AES_GCM_STEP, AES_GCM_STEP, 0, stream,
//       engine->gf_last4, engine->HL_sqr_long, engine->HH_sqr_long, AES_GCM_STEP * AES_GCM_STEP,
//       src, size / 16, engine->buffer1);
  dim3 numBlocks_1(AES_GCM_STEP);
  dim3 dimBlocks_1(AES_GCM_STEP);
  AES_GCM_mac_kernel<<<numBlocks_1, dimBlocks_1>>>(d_engine->gf_last4, d_engine->HL_sqr_long, 
  d_engine->HH_sqr_long, AES_GCM_STEP * AES_GCM_STEP, src, size / 16, d_engine->buffer1);


//   hipLaunchAddToBatch(batch, HIP_KERNEL_NAME(AES_GCM_mac_kernel), AES_GCM_STEP / 8, 8, 0, stream,
//       engine->gf_last4, engine->HL_long, engine->HH_long, AES_GCM_STEP,
//       engine->buffer1, AES_GCM_STEP * AES_GCM_STEP, engine->buffer2);
  dim3 numBlocks_2(AES_GCM_STEP / 8);
  dim3 dimBlocks_2(8);
  AES_GCM_mac_kernel<<<numBlocks_2, dimBlocks_2>>>(d_engine->gf_last4, d_engine->HL_long, 
  d_engine->HH_long, AES_GCM_STEP,
   d_engine->buffer1, AES_GCM_STEP * AES_GCM_STEP, d_engine->buffer2);


//   hipLaunchAddToBatch(batch, HIP_KERNEL_NAME(AES_GCM_mac_kernel), 1, 1, 0, stream,
//       engine->gf_last4, engine->HL, engine->HH, 1, engine->buffer2, AES_GCM_STEP, dst);
  dim3 numBlocks_3(1);
  dim3 dimBlocks_3(1);
  AES_GCM_mac_kernel<<<numBlocks_3, dimBlocks_3>>>(d_engine->gf_last4, d_engine->HL, d_engine->HH,
   1, d_engine->buffer2, AES_GCM_STEP, dst);


//   hipLaunchAddToBatch(batch, HIP_KERNEL_NAME(AES_GCM_mac_final_kernel), 1, 1, 0, stream,
//       engine->gf_last4, engine->HL, engine->HH, engine->sbox, engine->aes_roundkey,
//       nonce, dst, size, dst);
  dim3 numBlocks_4(1);
  dim3 dimBlocks_4(1);
  AES_GCM_mac_final_kernel<<<numBlocks_4, dimBlocks_4>>>(d_engine->gf_last4, d_engine->HL, 
  d_engine->HH, d_engine->sbox, d_engine->aes_roundkey, nonce, dst, size, dst);


}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

uint8_t key[AES_KEYLEN];
uint8_t nonce_host[crypto_aead_aes256gcm_NPUBBYTES];
uint8_t* nonce_device;
AES_GCM_engine* d_engine;
EVP_CIPHER_CTX* encrypt_ctx;
EVP_CIPHER_CTX* decrypt_ctx;


/*
template <typename... Args, typename F = void (*)(Args...)>
inline void hipLaunchNOW(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
                         std::uint32_t sharedMemBytes, hipStream_t stream, Args... args)
{
    hsa_kernel_dispatch_packet_t aql = {0};
    auto kernarg = hip_impl::make_kernarg(std::move(args)...);

    auto fun = hip_function_lookup((uintptr_t)kernel, stream);
    hip_function_to_aql(&aql, fun, DIM3_TO_AQL(numBlocks, dimBlocks), 0);

    __do_c_hipHccModuleLaunchKernel(&aql, stream, nullptr, (char *)kernarg.data(),
                                    kernarg.size(), nullptr, nullptr);
}
*/

void AES_GCM_init(const uint8_t* key) {
    //HIP_CHECK(hipMalloc(engine, sizeof(AES_GCM_engine)));
    gpuErrchk(cudaMalloc((void**)&d_engine, sizeof(AES_GCM_engine)));

    AES_GCM_engine h_engine;
    memcpy(h_engine.sbox, sbox_host, SBOX_SIZE);
    memcpy(h_engine.rsbox, rsbox_host, SBOX_SIZE);
    memcpy(h_engine.Rcon, Rcon_host, RCON_SIZE);
    memcpy(h_engine.key, key, AES_KEYLEN);

    cudaMemcpy(d_engine, &h_engine, sizeof(AES_GCM_engine), cudaMemcpyHostToDevice);

    dim3 numBlocks(1);
    dim3 dimBlocks(1);

    //hipLaunchNOW(HIP_KERNEL_NAME(AES_key_expansion_kernel), 1, 1, 0, stream,
    //    e->sbox, e->Rcon, e->key, e->aes_roundkey);
    AES_key_expansion_kernel<<<numBlocks, dimBlocks>>>(d_engine->sbox, d_engine->Rcon, d_engine->key, d_engine->aes_roundkey);
  
  
//   //HIP_CHECK(hipMemset(e->gcm_h, 0, 16)); // memset async isn't supported by HIP, so we block here
  AES_GCM_encrypt_one_block(d_engine->gcm_h);

//   HIP_CHECK(nw_hipMemcpySync(e->gf_last4, gf_last4_host, sizeof(gf_last4_host), hipMemcpyHostToDevice, stream));
    cudaMemcpy(d_engine->gf_last4, gf_last4_host, sizeof(gf_last4_host), cudaMemcpyHostToDevice);

//   hipLaunchNOW(HIP_KERNEL_NAME(AES_GCM_setup_gf_mult_table_kernel), 1, 1, 0, stream,
//       e->gf_last4, e->gcm_h, e->HL, e->HH, e->HL_long, e->HH_long, e->HL_sqr_long, e->HH_sqr_long);
AES_GCM_setup_gf_mult_table_kernel<<<numBlocks, dimBlocks>>>(d_engine->gf_last4, d_engine->gcm_h,
 d_engine->HL, d_engine->HH, d_engine->HL_long, d_engine->HH_long, d_engine->HL_sqr_long, d_engine->HH_sqr_long);
//   // no need to synchronize since all future GPU encryption operations will be on the same stream (hopefully?)
}

void AES_GCM_destroy(AES_GCM_engine* engine) {
  //HIP_CHECK(hipFree(engine));
}

// dst buffer should be of size: size + crypto_aead_aes256gcm_ABYTES
//void AES_GCM_encrypt(hip_launch_batch_t* batch, uint8_t* dst, const AES_GCM_engine* engine, const uint8_t* nonce,
//    const uint8_t* src, uint32_t size, hipStream_t stream) {
void AES_GCM_encrypt(uint8_t* dst, AES_GCM_engine* engine, uint8_t* nonce,
    const uint8_t* src, uint32_t size) {
//   assert(size % AES_BLOCKLEN == 0);
  AES_GCM_xcrypt(dst, nonce, src, size);
  AES_GCM_compute_mac(&dst[size], nonce, dst, size);
}

// src buffer should be of size: size + crypto_aead_aes256gcm_ABYTES
//void AES_GCM_decrypt(hip_launch_batch_t* batch, uint8_t* dst, const AES_GCM_engine* engine, const uint8_t* nonce,
//    const uint8_t* src, uint32_t size, hipStream_t stream) {
void AES_GCM_decrypt(uint8_t* dst, uint8_t* nonce,
    uint8_t* src, uint32_t size) {
//   assert(size % AES_BLOCKLEN == 0);
  AES_GCM_compute_mac(dst, nonce, src, size);
//   // TODO verify mac for i in crypto_aead_aes256gcm_ABYTES: (dst == src[size])
  AES_GCM_xcrypt(dst, nonce, src, size);
}

//void AES_GCM_next_nonce(hip_launch_batch_t* batch, uint8_t* nonce, hipStream_t stream) {
void AES_GCM_next_nonce(uint8_t* nonce) {
  //hipLaunchAddToBatch(batch, HIP_KERNEL_NAME(AES_GCM_next_nonce_kernel), 1, 1, 0, stream, nonce);
}


int main() {
    //assert(RAND_bytes(key, AES_KEYLEN) == 1);
    srand(0);
    for (int i = 0 ; i < AES_KEYLEN ; i++) {
        key[i] = i%255;
    }

  //code below is this: from lgm_memcpy.cpp
  /*
  EncryptionState::EncryptionState(hipStream_t stream) {
  assert(RAND_bytes(key, AES_KEYLEN) == 1);
  AES_GCM_init(&engine_device, key, stream);
  HIP_CHECK(hipMalloc(&nonce_device, crypto_aead_aes256gcm_NPUBBYTES));
  HIP_CHECK(nw_hipMemcpySync(nonce_device, nonce_host, crypto_aead_aes256gcm_NPUBBYTES,
      hipMemcpyHostToDevice, stream));
  encrypt_ctx = EVP_CIPHER_CTX_new();
  assert(encrypt_ctx != NULL);
  decrypt_ctx = EVP_CIPHER_CTX_new();
  assert(decrypt_ctx != NULL);
}
  */



    AES_GCM_init(key);
    cudaDeviceSynchronize();
    gpuErrchk(cudaMalloc(&nonce_device, crypto_aead_aes256gcm_NPUBBYTES));
    cudaMemcpy(nonce_device, nonce_host, crypto_aead_aes256gcm_NPUBBYTES,
        cudaMemcpyHostToDevice);
    // encrypt_ctx = EVP_CIPHER_CTX_new();
    // assert(encrypt_ctx != NULL);
    // decrypt_ctx = EVP_CIPHER_CTX_new();
    // assert(decrypt_ctx != NULL);


//translate this:
/*
void EncryptAsync(hip_launch_batch_t* batch, void* ciphertext, const void* plaintext, size_t sizeBytes,
    hipStream_t stream, EncryptionState& state) {
  AES_GCM_encrypt(batch, static_cast<uint8_t*>(ciphertext), state.engine_device,
      state.nonce_device, static_cast<const uint8_t*>(plaintext), sizeBytes, stream);
}
*/
// void EncryptAsync(void* ciphertext, const void* plaintext, size_t sizeBytes,
//     EncryptionState& state) {
//   AES_GCM_encrypt(static_cast<uint8_t*>(ciphertext), state.engine_device,
//       state.nonce_device, static_cast<const uint8_t*>(plaintext), sizeBytes);
// }

}