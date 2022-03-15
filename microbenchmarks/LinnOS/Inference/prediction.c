#include<stdio.h>
#include <stdbool.h> 
#include <time.h>
#include "weights.h"
#define LEN_INPUT 31
#define LEN_LAYER_0 256
#define LEN_LAYER_0_HALF 128
#define LEN_LAYER_1 2
#define FEAT_31

static bool prediction_model(long *input_vec_i, long *weight_0_T_ent, long *weight_1_T_ent, long *bias_0_ent, long *bias_1_ent) {

	long mid_res_i[LEN_LAYER_0], final_res_i[LEN_LAYER_1];
	int i, j, k, offset;

	for (j = 0, offset=0; j < LEN_LAYER_0; j++, offset+=LEN_INPUT) {
        mid_res_i[j] = 0;
        //loop unroll
		mid_res_i[j] += input_vec_i[0] * weight_0_T_ent[offset+0];
		mid_res_i[j] += input_vec_i[1] * weight_0_T_ent[offset+1];
		mid_res_i[j] += input_vec_i[2] * weight_0_T_ent[offset+2];
		mid_res_i[j] += input_vec_i[3] * weight_0_T_ent[offset+3];
		mid_res_i[j] += input_vec_i[4] * weight_0_T_ent[offset+4];
		mid_res_i[j] += input_vec_i[5] * weight_0_T_ent[offset+5];
		mid_res_i[j] += input_vec_i[6] * weight_0_T_ent[offset+6];
		mid_res_i[j] += input_vec_i[7] * weight_0_T_ent[offset+7];
		mid_res_i[j] += input_vec_i[8] * weight_0_T_ent[offset+8];
		mid_res_i[j] += input_vec_i[9] * weight_0_T_ent[offset+9];
		mid_res_i[j] += input_vec_i[10] * weight_0_T_ent[offset+10];
		mid_res_i[j] += input_vec_i[11] * weight_0_T_ent[offset+11];
		mid_res_i[j] += input_vec_i[12] * weight_0_T_ent[offset+12];
		mid_res_i[j] += input_vec_i[13] * weight_0_T_ent[offset+13];
		mid_res_i[j] += input_vec_i[14] * weight_0_T_ent[offset+14];
		mid_res_i[j] += input_vec_i[15] * weight_0_T_ent[offset+15];
		mid_res_i[j] += input_vec_i[16] * weight_0_T_ent[offset+16];
		mid_res_i[j] += input_vec_i[17] * weight_0_T_ent[offset+17];
		mid_res_i[j] += input_vec_i[18] * weight_0_T_ent[offset+18];
		mid_res_i[j] += input_vec_i[19] * weight_0_T_ent[offset+19];
		mid_res_i[j] += input_vec_i[20] * weight_0_T_ent[offset+20];
		mid_res_i[j] += input_vec_i[21] * weight_0_T_ent[offset+21];
		mid_res_i[j] += input_vec_i[22] * weight_0_T_ent[offset+22];
		mid_res_i[j] += input_vec_i[23] * weight_0_T_ent[offset+23];
		mid_res_i[j] += input_vec_i[24] * weight_0_T_ent[offset+24];
		mid_res_i[j] += input_vec_i[25] * weight_0_T_ent[offset+25];
		mid_res_i[j] += input_vec_i[26] * weight_0_T_ent[offset+26];
		mid_res_i[j] += input_vec_i[27] * weight_0_T_ent[offset+27];
		mid_res_i[j] += input_vec_i[28] * weight_0_T_ent[offset+28];
		mid_res_i[j] += input_vec_i[29] * weight_0_T_ent[offset+29];
		mid_res_i[j] += input_vec_i[30] * weight_0_T_ent[offset+30];

        // apply bias
        mid_res_i[j] += bias_0_ent[j];
        // relu
        if (mid_res_i[j] < 0) {
            mid_res_i[j] = 0;
        }
    }
    
    final_res_i[0] = 0;
    for(k=0; k<LEN_LAYER_0; k += 8) {
        final_res_i[0] += mid_res_i[k] * weight_1_T_ent[k];
		final_res_i[0] += mid_res_i[k+1] * weight_1_T_ent[k+1];
		final_res_i[0] += mid_res_i[k+2] * weight_1_T_ent[k+2];
		final_res_i[0] += mid_res_i[k+3] * weight_1_T_ent[k+3];
		final_res_i[0] += mid_res_i[k+4] * weight_1_T_ent[k+4];
		final_res_i[0] += mid_res_i[k+5] * weight_1_T_ent[k+5];
		final_res_i[0] += mid_res_i[k+6] * weight_1_T_ent[k+6];
		final_res_i[0] += mid_res_i[k+7] * weight_1_T_ent[k+7];
	}
	// apply bias
	final_res_i[0] += bias_1_ent[0];

	final_res_i[1] = 0;
    for(k=0; k<LEN_LAYER_0; k += 8) {
        final_res_i[1] += mid_res_i[k] * weight_1_T_ent[k+256];
		final_res_i[1] += mid_res_i[k+1] * weight_1_T_ent[k+257];
		final_res_i[1] += mid_res_i[k+2] * weight_1_T_ent[k+258];
		final_res_i[1] += mid_res_i[k+3] * weight_1_T_ent[k+259];
		final_res_i[1] += mid_res_i[k+4] * weight_1_T_ent[k+260];
		final_res_i[1] += mid_res_i[k+5] * weight_1_T_ent[k+261];
		final_res_i[1] += mid_res_i[k+6] * weight_1_T_ent[k+262];
		final_res_i[1] += mid_res_i[k+7] * weight_1_T_ent[k+263];
	}
	// apply bias
	final_res_i[1] += bias_1_ent[1];
    return final_res_i[0]>=(final_res_i[1])? false: true;
}

int main() {

	long feature_vec[31] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,9,0,0,0,9,0,0,0,9};

	clock_t start = clock();
	for(int i = 0; i < 1000; i++) {
		bool res = prediction_model(&feature_vec[0], &weight_i_0_T[0][0], &weight_i_1[0][0], bias_i_0, bias_i_1);
	}
	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("\n time taken : %f \n", seconds);
    
   return 0;
}

