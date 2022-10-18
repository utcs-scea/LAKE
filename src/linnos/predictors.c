#include <linux/delay.h>
#include <linux/ktime.h>
#include <linux/vmalloc.h>
#include <asm/fpu/api.h>
#include <linux/completion.h>
#include "predictors.h"
#include "variables.h"
#include "helpers.h"
#include "cuda.h"
#include "lake_shm.h"

int PREDICT_GPU_SYNC = 0;

//this is tough and rough
DEFINE_SPINLOCK(batch_lock);
#define _us 1000
//batch variables
const u64 window_size_ns = 400*_us;
const u32 max_batch_size = 32; //this cannot be more than 256 (allocated in main.c)
const u32 cpu_gpu_threshold = 4; //less than this we use cpu

u64 last_arrival_ns = 0;
u64 window_start_ns = 0;
u64 waiting = 0;
DECLARE_COMPLETION(batch_barrier);
//batch test variables
u32* window_size_hist; //allocated in main.c of kernel_hook, 128 elements
//GPU inference variables
struct GPU_weights gpu_weights[3]; //per-ssd weights, we are not going to have more than 3 ssds..
int use_cpu_instead = 0;
//this code has become too spaghetized, unfortunately
// we inherit from variables.h:
//	extern long *inputs_to_gpu;
//	extern long *gpu_outputs;
//	copy_inputs_to_gpu(u64 n_inputs) : copies from inputs_to_gpu
//	copy_results_from_gpu(u64 n_inputs) : copies to gpu_outputs.. but
//result is calculated using: gpu_outputs[i*32]>=(gpu_outputs[(i+1)32])? false: true;

//we need to hold the lock
static void gpu_batch_release(void) {
	window_size_hist[waiting] += 1;
	//let everyone go
	waiting = 0;
	pr_warn("completing..\n");
	complete_all(&batch_barrier);
	reinit_completion(&batch_barrier);
}

static int gpu_get_prediction(int idx) {
	return gpu_outputs[idx*32]>=(gpu_outputs[(idx+1)*32]) ?  false: true;
}

//hack: weights are actually device pointers here
void gpu_predict_batch(char *__feat_vec, int n_vecs, long **weights) {
	//do inference
	void *args[] = {
		&weights[0], &weights[2], &d_input_vec_i, &d_mid_res_i
	};
    check_error(cuLaunchKernel(batch_linnos_mid_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

    void *args1[] = {
		&weights[1], &weights[3], &d_mid_res_i, &d_final_res_i
	};

    check_error(cuLaunchKernel(batch_linnos_final_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				64, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args1, NULL),
			"cuLaunchKernel", __LINE__);
	if(PREDICT_GPU_SYNC == 1) {
		check_error(cuCtxSynchronize(), "cudaDeviceSynchronize", __LINE__);
	}
}

void do_gpu_inference(int n_vecs, long **weights) {
	copy_inputs_to_gpu(n_vecs);
	gpu_predict_batch(0, n_vecs, weights);
	copy_results_from_gpu(n_vecs);
}

//TODO: this assumes there's only one batch, when in fact
//we need one per device... 
// compare weight pointer to find which device we are in
// need three copies of every batch variable.

//this is what an IO calls when it calls predict()
bool gpu_batch_entry(char *feat_vec, int n_vecs, long **weights) {
	u64 my_id;
	bool is_last_arrival = false;
	bool my_prediction;
	u64 my_arrival;
	u32 err;
	unsigned long irqflags;

	// if someone gets stuck here, there's a batch going on
	//pr_warn("locking\n");
	spin_lock_irqsave(&batch_lock, irqflags);
	my_id = waiting++;
	my_arrival = ktime_get_ns();

	//if first, reset state
	if (unlikely(my_id == 0)) {
		pr_warn("***** FIRST, reseting state\n");
		window_start_ns = my_arrival;
	} 

	//I am the last, will release
	if(my_arrival - window_start_ns >= window_size_ns || waiting == max_batch_size) {
		pr_warn(" LAST, ts: last %llu  start %llu  dif %llu  window size %llu   waiting %llu\n", my_arrival, window_start_ns, my_arrival - window_start_ns, window_size_ns, waiting);

		//use cpu
		if(waiting < cpu_gpu_threshold) {
			use_cpu_instead = 1;
			pr_warn("setting use_cpu_instead to 1");
			//let ppl go then do our inference
			gpu_batch_release();
			spin_unlock_irqrestore(&batch_lock, irqflags);
			
			//XXX
			cpu_prediction_model(feat_vec, n_vecs, weights);
			return false;
		}
		//use GPU
		else {
			pr_warn(" #### running on GPU\n");
			use_cpu_instead = 0;
			//TODO
			//do_gpu_inference(waiting, gpu_weights[0].weights); //TODO: find this ssds index and use here
			//my_prediction = gpu_get_prediction(my_id);
			//XXX test
			for (err=0 ; err<32 ; err++)
			 	gpu_outputs[err] = false;
			gpu_batch_release();
			spin_unlock_irqrestore(&batch_lock, irqflags);
			return false;
		}
	}
	//first or middle
	else {
		spin_unlock_irqrestore(&batch_lock, irqflags);
		err = wait_for_completion_timeout(&batch_barrier, usecs_to_jiffies(window_size_ns/1000));
		if (err == 0) {  //if this was a timeout
			// if the first timed out, it needs to do the batch and release
			if (unlikely(my_id == 0)) {
				//first one timed out
				pr_warn("**** FIRST timed out: dif %llu  waiting %llu\n", my_arrival - window_start_ns, waiting);
				
				spin_lock_irqsave(&batch_lock, irqflags);
				//TODO inference
				gpu_batch_release();
				spin_unlock_irqrestore(&batch_lock, irqflags);
				return false;
			}
			//this case is tricky. a middle guy timed out, probably
			//because it hit the wait right after a batch was released
			else{
				pr_warn("******** bad: middle guy timed out\n");
				return cpu_prediction_model(feat_vec, n_vecs, weights);
			}
		}
		// if it wasnt a time out, we either get result or do cpu
		else{
			pr_warn("released, using cpu? %d\n", use_cpu_instead);
			if (use_cpu_instead) {
			 	cpu_prediction_model(feat_vec, n_vecs, weights);
				//XXX
				return false;
			}
			else
				return gpu_get_prediction(my_id);
		}
	}
}


bool fake_prediction_model(char *feat_vec, int n_vecs, long **weights) {
	return false;
}

bool cpu_prediction_model(char *feat_vec, int n_vecs, long **weights) {
	long input_vec_i[LEN_INPUT], mid_res_i[LEN_LAYER_0], final_res_i[LEN_LAYER_1];
	long *weight_0_T_ent, * bias_0_ent, *weight_1_T_ent, * bias_1_ent; 
	int i, j, k, offset;

	for (i=0 ; i<LEN_INPUT; i++) {
		input_vec_i[i] = (long)(feat_vec[i]);
	}

	weight_0_T_ent = weights[0];
	weight_1_T_ent = weights[1];
	bias_0_ent = weights[2];
	bias_1_ent = weights[3];

	for (j = 0, offset=0; j < LEN_LAYER_0; j++, offset+=LEN_INPUT) {
        mid_res_i[j] = 0;
        //loop unroll

		mid_res_i[j] += (input_vec_i[0] == 0 || weight_0_T_ent[offset+0] == 0)? 0 : input_vec_i[0] * weight_0_T_ent[offset+0];
		mid_res_i[j] += (input_vec_i[1] == 0 || weight_0_T_ent[offset+1] == 0)? 0 : input_vec_i[1] * weight_0_T_ent[offset+1];
		mid_res_i[j] += (input_vec_i[2] == 0 || weight_0_T_ent[offset+2] == 0)? 0 : input_vec_i[2] * weight_0_T_ent[offset+2];
		mid_res_i[j] += (input_vec_i[3] == 0 || weight_0_T_ent[offset+3] == 0)? 0 : input_vec_i[3] * weight_0_T_ent[offset+3];
		mid_res_i[j] += (input_vec_i[4] == 0 || weight_0_T_ent[offset+4] == 0)? 0 : input_vec_i[4] * weight_0_T_ent[offset+4];
		mid_res_i[j] += (input_vec_i[5] == 0 || weight_0_T_ent[offset+5] == 0)? 0 : input_vec_i[5] * weight_0_T_ent[offset+5];
		mid_res_i[j] += (input_vec_i[6] == 0 || weight_0_T_ent[offset+6] == 0)? 0 : input_vec_i[6] * weight_0_T_ent[offset+6];
		mid_res_i[j] += (input_vec_i[7] == 0 || weight_0_T_ent[offset+7] == 0)? 0 : input_vec_i[7] * weight_0_T_ent[offset+7];
		mid_res_i[j] += (input_vec_i[8] == 0 || weight_0_T_ent[offset+8] == 0)? 0 : input_vec_i[8] * weight_0_T_ent[offset+8];
		mid_res_i[j] += (input_vec_i[9] == 0 || weight_0_T_ent[offset+9] == 0)? 0 : input_vec_i[9] * weight_0_T_ent[offset+9];
		mid_res_i[j] += (input_vec_i[10] == 0 || weight_0_T_ent[offset+10] == 0)? 0 : input_vec_i[10] * weight_0_T_ent[offset+10];
		mid_res_i[j] += (input_vec_i[11] == 0 || weight_0_T_ent[offset+11] == 0)? 0 : input_vec_i[11] * weight_0_T_ent[offset+11];
		mid_res_i[j] += (input_vec_i[12] == 0 || weight_0_T_ent[offset+12] == 0)? 0 : input_vec_i[12] * weight_0_T_ent[offset+12];
		mid_res_i[j] += (input_vec_i[13] == 0 || weight_0_T_ent[offset+13] == 0)? 0 : input_vec_i[13] * weight_0_T_ent[offset+13];
		mid_res_i[j] += (input_vec_i[14] == 0 || weight_0_T_ent[offset+14] == 0)? 0 : input_vec_i[14] * weight_0_T_ent[offset+14];
		mid_res_i[j] += (input_vec_i[15] == 0 || weight_0_T_ent[offset+15] == 0)? 0 : input_vec_i[15] * weight_0_T_ent[offset+15];
		mid_res_i[j] += (input_vec_i[16] == 0 || weight_0_T_ent[offset+16] == 0)? 0 : input_vec_i[16] * weight_0_T_ent[offset+16];
		mid_res_i[j] += (input_vec_i[17] == 0 || weight_0_T_ent[offset+17] == 0)? 0 : input_vec_i[17] * weight_0_T_ent[offset+17];
		mid_res_i[j] += (input_vec_i[18] == 0 || weight_0_T_ent[offset+18] == 0)? 0 : input_vec_i[18] * weight_0_T_ent[offset+18];
		mid_res_i[j] += (input_vec_i[19] == 0 || weight_0_T_ent[offset+19] == 0)? 0 : input_vec_i[19] * weight_0_T_ent[offset+19];
		mid_res_i[j] += (input_vec_i[20] == 0 || weight_0_T_ent[offset+20] == 0)? 0 : input_vec_i[20] * weight_0_T_ent[offset+20];
		mid_res_i[j] += (input_vec_i[21] == 0 || weight_0_T_ent[offset+21] == 0)? 0 : input_vec_i[21] * weight_0_T_ent[offset+21];
		mid_res_i[j] += (input_vec_i[22] == 0 || weight_0_T_ent[offset+22] == 0)? 0 : input_vec_i[22] * weight_0_T_ent[offset+22];
		mid_res_i[j] += (input_vec_i[23] == 0 || weight_0_T_ent[offset+23] == 0)? 0 : input_vec_i[23] * weight_0_T_ent[offset+23];
		mid_res_i[j] += (input_vec_i[24] == 0 || weight_0_T_ent[offset+24] == 0)? 0 : input_vec_i[24] * weight_0_T_ent[offset+24];
		mid_res_i[j] += (input_vec_i[25] == 0 || weight_0_T_ent[offset+25] == 0)? 0 : input_vec_i[25] * weight_0_T_ent[offset+25];
		mid_res_i[j] += (input_vec_i[26] == 0 || weight_0_T_ent[offset+26] == 0)? 0 : input_vec_i[26] * weight_0_T_ent[offset+26];
		mid_res_i[j] += (input_vec_i[27] == 0 || weight_0_T_ent[offset+27] == 0)? 0 : input_vec_i[27] * weight_0_T_ent[offset+27];
		mid_res_i[j] += (input_vec_i[28] == 0 || weight_0_T_ent[offset+28] == 0)? 0 : input_vec_i[28] * weight_0_T_ent[offset+28];
		mid_res_i[j] += (input_vec_i[29] == 0 || weight_0_T_ent[offset+29] == 0)? 0 : input_vec_i[29] * weight_0_T_ent[offset+29];
		mid_res_i[j] += (input_vec_i[30] == 0 || weight_0_T_ent[offset+30] == 0)? 0 : input_vec_i[30] * weight_0_T_ent[offset+30];

        // apply bias
        mid_res_i[j] += bias_0_ent[j];
        // relu
        if (mid_res_i[j] < 0) {
            mid_res_i[j] = 0;
        }
    }
    
    final_res_i[0] = 0;
    for(k=0; k<LEN_LAYER_0; k += 8) {
        final_res_i[0] += (mid_res_i[k] == 0 || weight_1_T_ent[k] == 0)? 0 : mid_res_i[k] * weight_1_T_ent[k];
		final_res_i[0] += (mid_res_i[k+1] == 0 || weight_1_T_ent[k+1] == 0)? 0 : mid_res_i[k+1] * weight_1_T_ent[k+1];
		final_res_i[0] += (mid_res_i[k+2] == 0 || weight_1_T_ent[k+2] == 0)? 0 : mid_res_i[k+2] * weight_1_T_ent[k+2];
		final_res_i[0] += (mid_res_i[k+3] == 0 || weight_1_T_ent[k+3] == 0)? 0 : mid_res_i[k+3] * weight_1_T_ent[k+3];
		final_res_i[0] += (mid_res_i[k+4] == 0 || weight_1_T_ent[k+4] == 0)? 0 : mid_res_i[k+4] * weight_1_T_ent[k+4];
		final_res_i[0] += (mid_res_i[k+5] == 0 || weight_1_T_ent[k+5] == 0)? 0 : mid_res_i[k+5] * weight_1_T_ent[k+5];
		final_res_i[0] += (mid_res_i[k+6] == 0 || weight_1_T_ent[k+6] == 0)? 0 : mid_res_i[k+6] * weight_1_T_ent[k+6];
		final_res_i[0] += (mid_res_i[k+7] == 0 || weight_1_T_ent[k+7] == 0)? 0 : mid_res_i[k+7] * weight_1_T_ent[k+7];
	}
	// apply bias
	final_res_i[0] += bias_1_ent[0];

	final_res_i[1] = 0;
    for(k=0; k<LEN_LAYER_0; k += 8) {
        final_res_i[1] += (mid_res_i[k] == 0 || weight_1_T_ent[k+256] == 0)? 0 : mid_res_i[k] * weight_1_T_ent[k+256];
		final_res_i[1] += (mid_res_i[k+1] == 0 || weight_1_T_ent[k+257] == 0)? 0 : mid_res_i[k+1] * weight_1_T_ent[k+257];
		final_res_i[1] += (mid_res_i[k+2] == 0 || weight_1_T_ent[k+258] == 0)? 0 : mid_res_i[k+2] * weight_1_T_ent[k+258];
		final_res_i[1] += (mid_res_i[k+3] == 0 || weight_1_T_ent[k+259] == 0)? 0 : mid_res_i[k+3] * weight_1_T_ent[k+259];
		final_res_i[1] += (mid_res_i[k+4] == 0 || weight_1_T_ent[k+260] == 0)? 0 : mid_res_i[k+4] * weight_1_T_ent[k+260];
		final_res_i[1] += (mid_res_i[k+5] == 0 || weight_1_T_ent[k+261] == 0)? 0 : mid_res_i[k+5] * weight_1_T_ent[k+261];
		final_res_i[1] += (mid_res_i[k+6] == 0 || weight_1_T_ent[k+262] == 0)? 0 : mid_res_i[k+6] * weight_1_T_ent[k+262];
		final_res_i[1] += (mid_res_i[k+7] == 0 || weight_1_T_ent[k+263] == 0)? 0 : mid_res_i[k+7] * weight_1_T_ent[k+263];
	}
	// apply bias
	final_res_i[1] += bias_1_ent[1];

	// printk("Predictor returning %ld >= %ld = %d\n",
	// 	final_res_i[0], final_res_i[1],
	// 	(final_res_i[0] >= final_res_i[1]) ? false: true);
    return (final_res_i[0]>=final_res_i[1])? false: true;
}




void batch_release(void) {
	window_size_hist[waiting] += 1;
	waiting = 0;
	//realize execution here, including our data, and fill gpu_results
	//unblock everyone in this batch
	complete_all(&batch_barrier);
}

bool batch_test(char *feat_vec, int n_vecs, long **weights) {
	u64 my_id;
	bool is_last_arrival = false;
	u64 my_arrival;
	u32 err;

	spin_lock_irq(&batch_lock);
	my_arrival = ktime_get_ns();
	my_id = waiting++;
	//we are the first in this window
	if (my_id == 0)
		window_start_ns = my_arrival;

	//we are the last to arrive so far, lets account for possible out of order
	if(likely(my_arrival > last_arrival_ns))
		last_arrival_ns = my_arrival;

	//if more than window time passed or window is full
	if(last_arrival_ns - window_start_ns >= window_size_ns || waiting == max_batch_size) {
		is_last_arrival = true;
		batch_release();
	}
	spin_unlock_irq(&batch_lock);

	if (!is_last_arrival) {
		//if we are the first of this batch, we need to have a timeout
		if(my_id == 0) {
			//pr_warn("first\n");
			err = wait_for_completion_timeout(&batch_barrier, usecs_to_jiffies(window_size_ns/1000));
			if (err == 0) {  //this was a timeout
				spin_lock_irq(&batch_lock);
				batch_release();
				spin_unlock_irq(&batch_lock);
			}
		}
		else{ //if not first or last
			wait_for_completion(&batch_barrier);
		}
	}


	return false;
}