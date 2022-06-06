#include "pathfinder_cuda.h"

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )

#define NUM_ITR 4000

__global__ void dynproc_kernel(
                int iteration, 
                int *gpuWall,
                int *gpuSrc,
                int *gpuResults,
                int cols, 
                int rows,
                int startStep,
                int border)
{

        __shared__ int prev[PATHFINDER_BLOCK_SIZE];
        __shared__ int result[PATHFINDER_BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx=threadIdx.x;

        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data

        // calculate the small block size
    int small_block_cols = PATHFINDER_BLOCK_SIZE-iteration*HALO*2;

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkX = small_block_cols*bx-border;
        int blkXmax = blkX+PATHFINDER_BLOCK_SIZE-1;

        // calculate the global thread coordination
    int xidx = blkX+tx;

        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols-1) ? PATHFINDER_BLOCK_SIZE-1-(blkXmax-cols+1) : PATHFINDER_BLOCK_SIZE-1;

        int W = tx-1;
        int E = tx+1;

        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool isValid = IN_RANGE(tx, validXmin, validXmax);

    if(IN_RANGE(xidx, 0, cols-1)){
            prev[tx] = gpuSrc[xidx];
    }

        bool computed;
for (int k = 0; k < NUM_ITR; k++) {

        for (int i=0; i<iteration; i++) {
            computed = false;
            if( IN_RANGE(tx, i+1, PATHFINDER_BLOCK_SIZE-i-2) &&  \
                  isValid){
                  computed = true;
                  int left = prev[W];
                  int up = prev[tx];
                  int right = prev[E];
                  int shortest = MIN(left, up);
                  shortest = MIN(shortest, right);
                  int index = cols*(startStep+i)+xidx;
                  result[tx] = shortest + gpuWall[index];

            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)     //Assign the computation range
                prev[tx]= result[tx];
      }

      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if (computed){
          gpuResults[xidx]=result[tx];      
      }
}
}
