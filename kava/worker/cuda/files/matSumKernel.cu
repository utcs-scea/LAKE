// Vector addition (device code)

// size of the vectors to sum
#define N 128

extern "C" __global__ void matSum(int *a, int *b, int *c)
{
    int tid = blockIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}
