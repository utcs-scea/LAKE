#include<stdio.h>
#include <stdbool.h> 
#include "weights.h"
#include<sys/time.h>
#define LEN_INPUT 31
#define LEN_LAYER_0 256
#define LEN_LAYER_0_HALF 128
#define LEN_LAYER_1 2
#define FEAT_31
#define NUM_PARALLEL 16


__global__ void prediction_mid_layer(long *weight_0_T_ent, long *bias_0_ent, long *input_vec_i, long *mid_res_i) { 
	int j, offset;

	int threadId = threadIdx.x;
    int stride = blockDim.x;
	int input_ind = blockIdx.x*LEN_INPUT;
	int blockId = blockIdx.x;
	for (j = threadId, offset=threadId*LEN_INPUT; j < LEN_LAYER_0; j+=stride, offset+=LEN_INPUT*stride) {
		int update_index = blockId*stride + j;
        mid_res_i[update_index] = 0;
		//loop unroll
		mid_res_i[update_index] =  mid_res_i[update_index] + input_vec_i[input_ind + 0] * weight_0_T_ent[offset+0]
		+ input_vec_i[input_ind + 1] * weight_0_T_ent[offset+1]
		+ input_vec_i[input_ind + 2] * weight_0_T_ent[offset+2]
		+ input_vec_i[input_ind + 3] * weight_0_T_ent[offset+3]
		+ input_vec_i[input_ind + 4] * weight_0_T_ent[offset+4]
		+ input_vec_i[input_ind + 5] * weight_0_T_ent[offset+5]
		+ input_vec_i[input_ind + 6] * weight_0_T_ent[offset+6]
		+ input_vec_i[input_ind + 7] * weight_0_T_ent[offset+7]
		+ input_vec_i[input_ind + 8] * weight_0_T_ent[offset+8]
		+ input_vec_i[input_ind + 9] * weight_0_T_ent[offset+9]
		+ input_vec_i[input_ind + 10] * weight_0_T_ent[offset+10]
		+ input_vec_i[input_ind + 11] * weight_0_T_ent[offset+11]
		+ input_vec_i[input_ind + 12] * weight_0_T_ent[offset+12]
		+ input_vec_i[input_ind + 13] * weight_0_T_ent[offset+13]
		+ input_vec_i[input_ind + 14] * weight_0_T_ent[offset+14]
		+ input_vec_i[input_ind + 15] * weight_0_T_ent[offset+15]
		+ input_vec_i[input_ind + 16] * weight_0_T_ent[offset+16]
		+ input_vec_i[input_ind + 17] * weight_0_T_ent[offset+17]
		+ input_vec_i[input_ind+ 18] * weight_0_T_ent[offset+18]
		+ input_vec_i[input_ind + 19] * weight_0_T_ent[offset+19]
		+ input_vec_i[input_ind + 20] * weight_0_T_ent[offset+20]
		+ input_vec_i[input_ind + 21] * weight_0_T_ent[offset+21]
		+ input_vec_i[input_ind + 22] * weight_0_T_ent[offset+22]
		+ input_vec_i[input_ind + 23] * weight_0_T_ent[offset+23]
		+ input_vec_i[input_ind + 24] * weight_0_T_ent[offset+24]
		+ input_vec_i[input_ind + 25] * weight_0_T_ent[offset+25]
		+ input_vec_i[input_ind + 26] * weight_0_T_ent[offset+26]
		+ input_vec_i[input_ind + 27] * weight_0_T_ent[offset+27]
		+ input_vec_i[input_ind + 28] * weight_0_T_ent[offset+28]
		+ input_vec_i[input_ind + 29] * weight_0_T_ent[offset+29]
		+ input_vec_i[input_ind + 30] * weight_0_T_ent[offset+30];

        // apply bias
        mid_res_i[update_index] += bias_0_ent[threadId];
        // relu
        if (mid_res_i[update_index] < 0) {
            mid_res_i[update_index] = 0;
        }		
    }
}

__global__ void prediction_final_layer(long *weight_1_T_ent, long *bias_1_ent, long *mid_res_i, long *final_res_i) {
    
	int index = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
	int k;
	int update_index = index*dim + threadId;
	if (threadId < 32) {
		final_res_i[update_index] = 0;
		for(k = threadId; k<LEN_LAYER_0; k = k + 32) {
			final_res_i[update_index] =  final_res_i[update_index] + mid_res_i[index*LEN_LAYER_0 + k] * weight_1_T_ent[k];
		}
	} else {
		final_res_i[update_index] = 0;
		for(k = threadId - 32; k<LEN_LAYER_0; k = k + 32) {
			final_res_i[update_index] =  final_res_i[update_index] + mid_res_i[index*LEN_LAYER_0 + k] * weight_1_T_ent[k+256];
		}
	}
	__syncthreads();
	if (threadId == 0) {
		update_index = index*dim;
		for(int i = 1; i < 32; i++) {
			final_res_i[update_index] = final_res_i[update_index] + final_res_i[update_index + i];
		}
		final_res_i[update_index] =  final_res_i[update_index] + bias_1_ent[0];
	}
	if(threadId == 32) {
		update_index = index*dim + 32;
		for(int i = 1; i < 32; i++) {
			final_res_i[update_index] = final_res_i[update_index] + final_res_i[update_index + i];
		} 
		final_res_i[update_index] =  final_res_i[update_index] + bias_1_ent[1];
	}
}

static void prediction_model(long *d_input_vec_i, long *d_weight_0_T_ent, 
			long *d_weight_1_T_ent, long *d_bias_0_ent, long *d_bias_1_ent, long *d_mid_res_i, long *d_final_res_i, bool *res) {

	long final_res_i[NUM_PARALLEL*64];

	prediction_mid_layer<<<NUM_PARALLEL,256>>>(d_weight_0_T_ent, d_bias_0_ent, d_input_vec_i, d_mid_res_i);
	prediction_final_layer<<<NUM_PARALLEL, 64>>>(d_weight_1_T_ent, d_bias_1_ent, d_mid_res_i, d_final_res_i);

	cudaMemcpy(final_res_i, d_final_res_i, sizeof(long) * 64 * NUM_PARALLEL, cudaMemcpyDeviceToHost);
	for(int i = 0; i < NUM_PARALLEL; i++) {
		// printf("\n %ld", final_res_i[i*64]);
		// printf("\n %ld", final_res_i[i*64 + 32]);
		res[i] = final_res_i[i*64]>=(final_res_i[i *64 + 32])? false: true;
	}
	
}

int main() {
	long *weight_0_T_ent, * bias_0_ent, *weight_1_T_ent, * bias_1_ent; 
	long input_vec_i[31] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,9,0,0,0,9,0,0,0,9};
	long parallel_input[NUM_PARALLEL][31];
	for(int i = 0 ; i < NUM_PARALLEL; i++) {
		for(int j = 0; j < 31; j++)
			parallel_input[i][j] = input_vec_i[j];
	}

	weight_0_T_ent = &weight_i_0_T[0][0];
	weight_1_T_ent = &weight_i_1[0][0];
	bias_0_ent = bias_i_0;
	bias_1_ent = bias_i_1;

	long *d_weight_0_T_ent, *d_weight_1_T_ent, *d_bias_0_ent, *d_bias_1_ent, *d_input_vec_i, *d_mid_res_i, *d_final_res_i;

	
	cudaMalloc((void**)&d_weight_0_T_ent, sizeof(long) * 256*31);
	cudaMalloc((void**)&d_weight_1_T_ent, sizeof(long) * 256*2);
	cudaMalloc((void**)&d_bias_0_ent, sizeof(long) * 256);
	cudaMalloc((void**)&d_bias_1_ent, sizeof(long) *2);

	cudaMalloc((void**)&d_mid_res_i, sizeof(long) *LEN_LAYER_0 * NUM_PARALLEL);
	cudaMalloc((void**)&d_final_res_i, sizeof(long) *LEN_LAYER_1 * NUM_PARALLEL *32);
	bool res[NUM_PARALLEL];

	
	cudaMalloc((void**)&d_input_vec_i, sizeof(long) *LEN_INPUT * NUM_PARALLEL);
	cudaMemcpy(d_weight_0_T_ent, weight_0_T_ent, sizeof(long) * 256*31, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight_1_T_ent, weight_1_T_ent, sizeof(long) * 256*2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bias_0_ent, bias_0_ent, sizeof(long) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bias_1_ent, bias_1_ent, sizeof(long) * 2, cudaMemcpyHostToDevice);
	struct timeval tval_before, tval_after, tval_result;
	gettimeofday(&tval_before, NULL);
	
	for(int i = 0; i < 1; i++) {
		cudaMemcpy(d_input_vec_i, parallel_input, sizeof(long) * LEN_INPUT * NUM_PARALLEL, cudaMemcpyHostToDevice);
		 prediction_model(d_input_vec_i, d_weight_0_T_ent, 
			d_weight_1_T_ent, d_bias_0_ent, d_bias_1_ent, d_mid_res_i, d_final_res_i, res);
	}
	
	gettimeofday(&tval_after, NULL);
	timersub(&tval_after, &tval_before, &tval_result);
	printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

	cudaFree(d_input_vec_i);
	cudaFree(d_weight_0_T_ent);
	cudaFree(d_weight_1_T_ent);
	cudaFree(d_bias_0_ent);
	cudaFree(d_bias_1_ent);
	cudaFree(d_mid_res_i);
	cudaFree(d_final_res_i);

	

	
		
   return 0;
}

