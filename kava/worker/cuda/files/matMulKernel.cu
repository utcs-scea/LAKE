/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>

#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]

#define block_size 16

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void matrixMul(int *C, int *A, int *B, size_t wA,
                                     size_t wB) {
  // Block index
  size_t bx = blockIdx.x;
  size_t by = blockIdx.y;

  // Thread index
  size_t tx = threadIdx.x;
  size_t ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  size_t aBegin = wA * block_size * by;

  // Index of the last sub-matrix of A processed by the block
  size_t aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  size_t aStep = block_size;

  // Index of the first sub-matrix of B processed by the block
  size_t bBegin = block_size * bx;

  // Step size used to iterate through the sub-matrices of B
  size_t bStep = block_size * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  int Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (size_t a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ int As[block_size][block_size];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ int Bs[block_size][block_size];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    AS(ty, tx) = A[a + wA * ty + tx];
    BS(ty, tx) = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll

    for (size_t k = 0; k < block_size; ++k) Csub += AS(ty, k) * BS(k, tx);

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  size_t c = wB * block_size * by + block_size * bx;
  C[c + wB * ty + tx] = Csub;
}

#endif  // #ifndef _MATRIXMUL_KERNEL_H_
