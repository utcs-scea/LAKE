
__global__ void hello_kernel(int* inputs, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        inputs[id] = id;
    }
}