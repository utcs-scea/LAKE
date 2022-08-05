#include "gcm.h"

// This function produces Nb(Nr+1) round keys. The round keys are used in each round to decrypt the states.
__device__ void KeyExpansion(uint8_t* sbox, uint8_t* Rcon, uint8_t* RoundKey,
    uint8_t* Key)
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
        uint8_t u8tmp = tempa[0];
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
__device__ void AddRoundKey(uint8_t r, state_t* state, uint8_t* RoundKey)
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
__device__ void SubBytes(uint8_t* sbox, state_t* state)
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
__device__ void Cipher(uint8_t* sbox, state_t* state, uint8_t* RoundKey)
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

__global__ void AES_key_expansion_kernel(uint8_t* sbox, uint8_t* Rcon,
    uint8_t* key, uint8_t* roundkey) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    KeyExpansion(sbox, Rcon, roundkey, key);
  }
}

__global__ void AES_encrypt_one_block_kernel(uint8_t* sbox, uint8_t* roundkey,
    uint8_t* data) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    Cipher(sbox, (state_t*)data, roundkey);
  }
}

__device__ uint32_t get_be32(uint8_t *a)
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

__device__ void gf_mult_fast(uint64_t* last4, uint64_t* HL, uint64_t* HH,
    uint8_t* x, uint8_t* output) {
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

__device__ void gf_build_table(uint8_t* h, uint64_t* HL, uint64_t* HH) {
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
    uint64_t* last4, uint8_t* h, uint64_t* HL, uint64_t* HH,
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

__global__ void AES_GCM_xcrypt_kernel(uint8_t* dst, uint8_t* sbox, uint8_t* roundkey,
    uint8_t* nonce, uint8_t* src, uint32_t size) {
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
    uint64_t* last4, uint64_t* HL, uint64_t* HH, uint8_t* sbox, uint8_t* roundkey,
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