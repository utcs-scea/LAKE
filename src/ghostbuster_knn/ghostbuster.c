#ifdef __KERNEL__
#include <linux/module.h>
#include <linux/random.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <asm/uaccess.h>
#include <linux/delay.h>
#include <linux/ktime.h>
#include "cuda.h"
#include "lake_shm.h"
#define PRINT(...) pr_warn(__VA_ARGS__)
#else
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <errno.h>
static inline uint64_t get_tsns() {
    struct timeval current_time;
    gettimeofday(&current_time, 0);
    return current_time.tv_sec*1000000000 + current_time.tv_usec*1000;
}
void get_random_bytes(char* x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = rand();
}

#define usleep_range(X,Y) sleep(X/1000000)
#define ktime_get_ns() get_tsns()
#define u64 uint64_t
#define vmalloc(X) malloc(X)
#define vfree(X) free((void *)X)
#define kava_alloc(...) malloc(__VA_ARGS__)
#define kava_free(...) free(__VA_ARGS__)
#define PRINT(...) printf(__VA_ARGS__)
#include <cuda.h>
#endif

static char *cubin_path = "/home/hfingler/hf-HACK/src/ghostbuster_knn/knncuda.cubin";
#ifdef __KERNEL__
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to .cubin");
#endif

#define BLOCK_DIM 16
#define WARMS 2
#define RUNS 5

// XXX Need to handle FLOATs eventually
typedef int FLOAT;
typedef u64 DOUBLE;

struct cuda_ctx
{
    CUdevice dev;
    CUcontext ctx;
    CUstream stream;
    CUmodule mod;
    CUfunction compute_dist;
    CUfunction modified_insertion_sort;
    CUfunction compute_sqrt;
};

typedef struct
{
    u64 x;
    u64 y;
    u64 z;
} dim3;

// CUDA vars
struct cuda_ctx ctx;

// KNN vars
static int ref_nb = 16384;
//static int ref_nb = 2048;
static int query_nb = 4096;

int init_cuda(void)
{
    int ret = 0;

    ret = cuInit(0);
    if (ret) {
        PRINT("Err cuInit %d\n", ret);
        goto out;
    }

    ret = cuDeviceGet(&ctx.dev, 0);
    if (ret) {
        PRINT("Err cuDeviceGet %d\n", ret);
        goto out;
    }

    ret = cuCtxCreate(&ctx.ctx, 0, ctx.dev);
    if (ret) {
        PRINT("Err cuCtxCreate %d\n", ret);
        goto out;
    }

    ret = cuModuleLoad(&ctx.mod, cubin_path);
    if (ret) {
        PRINT("Err cuModuleLoad %d\n", ret);
        goto out;
    }

    ret = cuModuleGetFunction(&ctx.compute_dist, ctx.mod,
                               "_Z17compute_distancesPfiiS_iiiS_");
    if (ret) {
        PRINT("Err cuModuleGetFunction %d\n", ret);
        goto out;
    }

    ret = cuModuleGetFunction(&ctx.modified_insertion_sort, ctx.mod,
                               "_Z23modified_insertion_sortPfiPiiiii");
    if (ret) {
        PRINT("Err cuModuleGetFunction 2 %d\n", ret);
        goto out;
    }

    ret = cuModuleGetFunction(&ctx.compute_sqrt, ctx.mod,
                               "_Z12compute_sqrtPfiii");
    if (ret) {
        PRINT("Err cuModuleGetFunction 3 %d\n", ret);
        goto out;
    }

out: 
    return ret;
}
// ==================== End CUDA ====================

// ==================== Start KNN ====================
void initialize_data(FLOAT *ref, int ref_nb, FLOAT *query,
                     int query_nb, int dim)
{
    int i;
    int rand;
    // XXX Resolve floats
    // Generate random reference points
    for (i = 0; i < ref_nb * dim; ++i) {
        get_random_bytes(&rand, sizeof(rand));
        ref[i] = 10 * (FLOAT)(rand); // / (DOUBLE) RAND_MAX );
    }
    // Generate random query points
    for (i = 0; i < query_nb * dim; ++i) {
        get_random_bytes(&rand, sizeof(rand));
        query[i] = 10 * (FLOAT)(rand); // / (DOUBLE) RAND_MAX );
    }
}

static u64 ctime, ttime;

int knn_cuda( const FLOAT *ref, int ref_nb, const FLOAT *query,
              int query_nb, int dim, FLOAT *knn_dist,
              int *knn_index, int measure_compute)
{
    int ret = 0;
    // Launch params
    dim3 block0, block1, block2;
    dim3 grid0, grid1, grid2;
    // Vars for computation
    CUdeviceptr ref_dev, query_dev, dist_dev, index_dev;
    size_t ref_pitch_in_bytes;
    size_t query_pitch_in_bytes;
    size_t dist_pitch_in_bytes;
    size_t index_pitch_in_bytes;
    // Pitch values
    size_t ref_pitch, query_pitch;
    size_t dist_pitch, index_pitch;
    // Params for pitch (4, 8, or 16)
    size_t element_size_bytes = 16;
    u64 t_start, t_stop, c_start, c_stop;
    ctime = 0; ttime = 0; 

    t_start = ktime_get_ns();
    // Allocate global memory
    ret |= cuMemAllocPitch( &ref_dev, &ref_pitch_in_bytes,
                            ref_nb * sizeof( FLOAT ), dim, element_size_bytes );
    ret |= cuMemAllocPitch( &query_dev, &query_pitch_in_bytes,
                            query_nb * sizeof( FLOAT ), dim, element_size_bytes );
    ret |= cuMemAllocPitch( &dist_dev, &dist_pitch_in_bytes,
                            query_nb * sizeof( FLOAT ),
                            ref_nb, element_size_bytes );
    ret |= cuMemAllocPitch( &index_dev, &index_pitch_in_bytes,
                            query_nb * sizeof( int ), dim, element_size_bytes );
    if (ret) {
        PRINT( "Memory allocation error\n" );
        goto out;
    }

    // Deduce pitch values
    ref_pitch = ref_pitch_in_bytes / sizeof( FLOAT );
    query_pitch = query_pitch_in_bytes / sizeof( FLOAT );
    dist_pitch = dist_pitch_in_bytes / sizeof( FLOAT );
    index_pitch = index_pitch_in_bytes / sizeof( int );

    // Check pitch values
    if (query_pitch != dist_pitch || query_pitch != index_pitch ) {
        PRINT("Invalid pitch value\n" );
        goto out;
    }

    // Copy reference and query data from the host to the device
    ret |= cuMemcpyHtoDAsync(ref_dev, ref, ref_pitch_in_bytes, 0);
    ret |= cuMemcpyHtoDAsync(query_dev, query, query_pitch_in_bytes, 0);
    if (ret) {
        PRINT( "Unable to copy data from host to device\n" );
        goto out;
    }

    //if we're measuring just compute, wait until everything is done
    if (measure_compute) {
        cuCtxSynchronize();
        c_start = ktime_get_ns();
    }
    
    // Compute the squared Euclidean distances
    block0 = (dim3) { BLOCK_DIM, BLOCK_DIM, 1 };
    grid0 = (dim3) { query_nb / BLOCK_DIM, ref_nb / BLOCK_DIM, 1 };
    if (query_nb % BLOCK_DIM != 0) {
        grid0.x += 1;
    }
    if (ref_nb % BLOCK_DIM != 0) {
        grid0.y += 1;
    }

    void *args0[] = { &ref_dev, &ref_nb, &ref_pitch,
                    &query_dev, &query_nb, &query_pitch,
                    &dim, &dist_dev };
    cuLaunchKernel( ctx.compute_dist, grid0.x, grid0.y,
                    grid0.z, block0.x, block0.y,
                    block0.z, 0, 0,
                    args0, NULL);

    // Sort the distances with their respective indexes
    block1 = (dim3) { 256, 1, 1 };
    grid1 = (dim3) { query_nb / 256, 1, 1 };
    if (query_nb % 256 != 0) 
        grid1.x += 1;
  
    void *args1[] = { &dist_dev, &dist_pitch, &index_dev,
                    &index_pitch, &query_nb, &ref_nb,
                    &dim };
    cuLaunchKernel( ctx.modified_insertion_sort, grid1.x, grid1.y,
                    grid1.z, block1.x, block1.y,
                    block1.z, 0, 0,
                    args1, NULL);

    // Compute the square root of the k smallest distances
    block2 = (dim3) { 16, 16, 1 };
    grid2 = (dim3) { query_nb / 16, dim / 16, 1 };
    if ( query_nb % 16 != 0 ) {
        grid2.x += 1;
    }
    if ( dim % 16 != 0 ) {
        grid2.y += 1;
    }
    void *args2[] = { &dist_dev, &query_nb, &query_pitch, &dim };
    cuLaunchKernel( ctx.compute_sqrt, grid2.x, grid2.y,
                    grid2.z, block2.x, block2.y,
                    block2.z, 0, 0,
                    args2, NULL);

    //if we're measuring just compute, wait until everything is done
    if (measure_compute) {
        cuCtxSynchronize();
        c_stop = ktime_get_ns();
        ctime = c_stop - c_start;
    }

    // Copy k smallest distances / indexes from the device to the host
    ret |= cuMemcpyDtoHAsync( knn_dist, dist_dev,
                            dist_pitch_in_bytes, 0 );
    ret |= cuMemcpyDtoHAsync( knn_index, index_dev,
                            index_pitch_in_bytes, 0 );
    if ((ret = cuCtxSynchronize())) {
        PRINT( "Unable to execute modified_insertion_sort kernel\n" );
        goto out;
    }

    t_stop = ktime_get_ns();
    ttime = t_stop - t_start;
out:
    cuMemFree( ref_dev );
    cuMemFree( query_dev );
    cuMemFree( dist_dev );
    cuMemFree( index_dev ); 
    return ret;
}

// XXX Should time at some point
int test(const FLOAT *ref, int ref_nb, const FLOAT *query, int query_nb,
         int dim)
{
    int ret = 0;
    int i, measure_comp;
    int *test_knn_index;
    FLOAT *test_knn_dist;
    u64 ctimes;
    u64 ttimes;
    // Allocate memory for computed k-NN neighbors
    test_knn_dist = (FLOAT *)kava_alloc(query_nb * sizeof(FLOAT));
    test_knn_index = (int *)kava_alloc(query_nb * sizeof(int));

    // Allocation check
    if (!test_knn_dist || !test_knn_index) {
        PRINT("Error allocating CPU memory for KNN results\n");
        ret = -ENOMEM;
        goto out;
    }

    usleep_range(200, 500);
    ctimes = 0;
    ttimes = 0;
    
    for (measure_comp = 0; measure_comp < 2; ++measure_comp) {
        for (i = 0; i < WARMS + RUNS; ++i) {
            if ((ret = knn_cuda(ref, ref_nb, query, query_nb, dim,
                        test_knn_dist, test_knn_index, measure_comp))) {
                PRINT("Computation failed on round %d\n", i);
                goto out;
            }

            if (i >= WARMS) {
                if (measure_comp == 1)
                    ctimes += ctime;
                else
                    ttimes += ttime;
            }
            usleep_range(2000, 5000);
        }
    }
    PRINT("gpu_%d, %lld, %lld\n", dim, ctimes / (RUNS * 1000), ttimes / (RUNS * 1000));

out:
    kava_free(test_knn_dist);
    kava_free(test_knn_index);

    return ret;
}

// Allocate input points and output k-NN distances / indexes
int run_knn(void)
{
    int ret = 0;
    FLOAT *ref;
    FLOAT *query;
    int ref_sz;
    int query_sz;
    int i, dim;
    int dims[] = {1,2,4,8, 16, 32, 64, 128,256,512,1024};
    int ndims = 11;
    //int dims[] = {256,512,1024};
    //int ndims = 3;

    for (i = 0; i < ndims; i++) {
        dim = dims[i];
        ref_sz = ref_nb * dim * sizeof(FLOAT);
        query_sz = query_nb * dim * sizeof(FLOAT);
        ref = (FLOAT *)kava_alloc(ref_sz);
        query = (FLOAT *)kava_alloc(query_sz);

        // Allocation checks
        if (!ref || !query) {
            PRINT("Error allocating KNN CPU resources\n");
            ret = -ENOMEM;
            goto out;
        }
        // Initialize reference and query points with random values
        initialize_data(ref, ref_nb, query, query_nb, dim);
        ret = test(ref, ref_nb, query, query_nb, dim);
        if (ret) {
            PRINT("KNN execution test failed\n");
            ret = -ENOENT;
            goto out;
        }
out:
        kava_free(ref);
        kava_free(query);
    }

    return 0;
}
// ==================== End KNN ====================

#ifdef __KERNEL__
static int __init ghost_buster_init(void)
{
    int ret = 0;
    if ((ret = init_cuda())) {
        PRINT("error init cuda");
        return ret;
    }
    if ((ret = run_knn())) {
        return ret;
    }
    return ret;
}

static void __exit ghost_buster_fini(void)
{
}

module_init(ghost_buster_init);
module_exit(ghost_buster_fini);

MODULE_AUTHOR("Ariel Szekely");
MODULE_DESCRIPTION("A module to detect Spectre attacks"
                   "(aka a Ghost Buster... get it?)");
MODULE_LICENSE("GPL");
MODULE_VERSION(
    __stringify(1) "." __stringify(0) "." __stringify(0) "."
                                                         "0");

#else

int main() {
    int ret = 0;
    if ((ret = init_cuda())) {
        return ret;
    }
    if ((ret = run_knn())) {
        return ret;
    }
    return ret;
}

#endif