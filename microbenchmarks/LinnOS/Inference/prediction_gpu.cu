#include <stdio.h>
#include <stdbool.h> 
#include "weights.h"
#include "kernels.h"


__global__ void prediction_mid_layer_naive(long *weight_0_T_ent, long *bias_0_ent, long *input_vec_i, long *mid_res_i) { 
	int j, offset;

	int index = threadIdx.x;
    int stride = blockDim.x;

	for (j = index, offset=j*LEN_INPUT; j < LEN_LAYER_0; j+=stride, offset+=LEN_INPUT*stride) {
        mid_res_i[j] = 0;
        //loop unroll
		mid_res_i[j] =  mid_res_i[j] + input_vec_i[0] * weight_0_T_ent[offset+0]
		+ input_vec_i[1] * weight_0_T_ent[offset+1]
		+ input_vec_i[2] * weight_0_T_ent[offset+2]
		+ input_vec_i[3] * weight_0_T_ent[offset+3]
		+ input_vec_i[4] * weight_0_T_ent[offset+4]
		+ input_vec_i[5] * weight_0_T_ent[offset+5]
		+ input_vec_i[6] * weight_0_T_ent[offset+6]
		+ input_vec_i[7] * weight_0_T_ent[offset+7]
		+ input_vec_i[8] * weight_0_T_ent[offset+8]
		+ input_vec_i[9] * weight_0_T_ent[offset+9]
		+ input_vec_i[10] * weight_0_T_ent[offset+10]
		+ input_vec_i[11] * weight_0_T_ent[offset+11]
		+ input_vec_i[12] * weight_0_T_ent[offset+12]
		+ input_vec_i[13] * weight_0_T_ent[offset+13]
		+ input_vec_i[14] * weight_0_T_ent[offset+14]
		+ input_vec_i[15] * weight_0_T_ent[offset+15]
		+ input_vec_i[16] * weight_0_T_ent[offset+16]
		+ input_vec_i[17] * weight_0_T_ent[offset+17]
		+ input_vec_i[18] * weight_0_T_ent[offset+18]
		+ input_vec_i[19] * weight_0_T_ent[offset+19]
		+ input_vec_i[20] * weight_0_T_ent[offset+20]
		+ input_vec_i[21] * weight_0_T_ent[offset+21]
		+ input_vec_i[22] * weight_0_T_ent[offset+22]
		+ input_vec_i[23] * weight_0_T_ent[offset+23]
		+ input_vec_i[24] * weight_0_T_ent[offset+24]
		+ input_vec_i[25] * weight_0_T_ent[offset+25]
		+ input_vec_i[26] * weight_0_T_ent[offset+26]
		+ input_vec_i[27] * weight_0_T_ent[offset+27]
		+ input_vec_i[28] * weight_0_T_ent[offset+28]
		+ input_vec_i[29] * weight_0_T_ent[offset+29]
		+ input_vec_i[30] * weight_0_T_ent[offset+30];

        // apply bias
        mid_res_i[j] += bias_0_ent[j];
        // relu
        if (mid_res_i[j] < 0) {
            mid_res_i[j] = 0;
        }
    }
}

__global__ void prediction_final_layer_naive(long *weight_1_T_ent, long *bias_1_ent, long *mid_res_i, long *final_res_i) {
    final_res_i[0] = 0;
	int k;
    for(k=0; k<LEN_LAYER_0; k ++) {
        final_res_i[0] =  final_res_i[0] + mid_res_i[k] * weight_1_T_ent[k];
	}
	// apply bias
	final_res_i[0] =  final_res_i[0] + bias_1_ent[0];

	final_res_i[1] = 0;
    for(k=0; k<LEN_LAYER_0; k ++) {
        final_res_i[1] =  final_res_i[1] + mid_res_i[k] * weight_1_T_ent[k+256];
	}
	// apply bias
	final_res_i[1] =  final_res_i[1] + bias_1_ent[1];
}

void prediction_model(long *d_input_vec_i, long *d_weight_0_T_ent, 
			long *d_weight_1_T_ent, long *d_bias_0_ent, long *d_bias_1_ent, long *d_mid_res_i, long *d_final_res_i) {
	prediction_mid_layer_naive<<<1,256>>>(d_weight_0_T_ent, d_bias_0_ent, d_input_vec_i, d_mid_res_i);
	prediction_final_layer_naive<<<1,1>>>(d_weight_1_T_ent, d_bias_1_ent, d_mid_res_i, d_final_res_i);
	cudaDeviceSynchronize();
}

static long final_res_i[LEN_LAYER_1];
static long *weight_0_T_ent, * bias_0_ent, *weight_1_T_ent, * bias_1_ent; 
static long input_vec_i[31] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,9,0,0,0,9,0,0,0,9};
static long *d_weight_0_T_ent, *d_weight_1_T_ent, *d_bias_0_ent, *d_bias_1_ent, *d_input_vec_i, *d_mid_res_i, *d_final_res_i;


bool get_result_naive() {
	cudaMemcpy(final_res_i, d_final_res_i, sizeof(long) * 2, cudaMemcpyDeviceToHost);
	return final_res_i[0]>=(final_res_i[1])? false: true;
}

void setup_naive() {
	weight_0_T_ent = &weight_i_0_T[0][0];
	weight_1_T_ent = &weight_i_1[0][0];
	bias_0_ent = bias_i_0;
	bias_1_ent = bias_i_1;

	cudaMalloc((void**)&d_input_vec_i, sizeof(long) *LEN_INPUT);
	cudaMalloc((void**)&d_weight_0_T_ent, sizeof(long) * 256*31);
	cudaMalloc((void**)&d_weight_1_T_ent, sizeof(long) * 256*2);
	cudaMalloc((void**)&d_bias_0_ent, sizeof(long) * 256);
	cudaMalloc((void**)&d_bias_1_ent, sizeof(long) *2);

	cudaMemcpy(d_weight_0_T_ent, weight_0_T_ent, sizeof(long) * 256*31, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight_1_T_ent, weight_1_T_ent, sizeof(long) * 256*2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bias_0_ent, bias_0_ent, sizeof(long) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bias_1_ent, bias_1_ent, sizeof(long) * 2, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_mid_res_i, sizeof(long) *LEN_LAYER_0);
	cudaMalloc((void**)&d_final_res_i, sizeof(long) *LEN_LAYER_1);

}

void copy_inputs_naive() {
	cudaMemcpy(d_input_vec_i, input_vec_i, sizeof(long) * LEN_INPUT, cudaMemcpyHostToDevice);
}

	
void infer_naive() {
	prediction_model(d_input_vec_i, d_weight_0_T_ent, 
			d_weight_1_T_ent, d_bias_0_ent, d_bias_1_ent, d_mid_res_i, d_final_res_i);
}
	
void clean_naive() {
	cudaFree(d_input_vec_i);
	cudaFree(d_weight_0_T_ent);
	cudaFree(d_weight_1_T_ent);
	cudaFree(d_bias_0_ent);
	cudaFree(d_bias_1_ent);
	cudaFree(d_mid_res_i);
	cudaFree(d_final_res_i);
}