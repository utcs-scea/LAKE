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
#define _us 1000
//batch variables
const u64 window_size_ns = 300*_us;
const u32 max_batch_size = 16; //this cannot be more than 256 (allocated in main.c)
const u32 cpu_gpu_threshold = 8; //less than this we use cpu

//batch test variables
u32* window_size_hist; //allocated in main.c of kernel_hook, 128 elements
u32 n_used_gpu = 0;

#define NUMBER_DEVICES 1

///DECLARE_COMPLETION(batch_completed);
///DECLARE_COMPLETION(batch_block);
///DECLARE_COMPLETION(finalize_batch);
struct completion batch_completed[NUMBER_DEVICES];
struct completion batch_block[NUMBER_DEVICES];
struct completion finalize_batch[NUMBER_DEVICES];
//DEFINE_SPINLOCK(batch_running_lock);
spinlock_t batch_running_lock[NUMBER_DEVICES];
//DEFINE_SPINLOCK(batch_exited);
spinlock_t batch_exited[NUMBER_DEVICES];
spinlock_t batch_lock[NUMBER_DEVICES];

int is_batch_running[NUMBER_DEVICES];
int n_exited[NUMBER_DEVICES];
int this_batch_size[NUMBER_DEVICES];
int pending[NUMBER_DEVICES];
u64 window_start_ns[NUMBER_DEVICES];
u64 waiting[NUMBER_DEVICES];

//GPU inference variables
struct GPU_weights gpu_weights[NUMBER_DEVICES]; //per-ssd weights, we are not going to have more than NUMBER_DEVICES ssds..
int use_cpu_instead[NUMBER_DEVICES];
int is_batch_blocked[NUMBER_DEVICES];

void predictors_mgpu_init(void) {
	int i;
	for (i=0 ; i < NUMBER_DEVICES ; i++) {
		is_batch_running[i] = 0;
		n_exited[i] = 0;
		this_batch_size[i] = 0;
		pending[i] = 0;
		window_start_ns[i] = 0;
		waiting[i] = 0;
		use_cpu_instead[i] = 0;
		is_batch_blocked[i] = 0;

		init_completion(&batch_completed[i]);
		init_completion(&batch_block[i]);
		init_completion(&finalize_batch[i]);
		spin_lock_init(&batch_running_lock[i]);
		spin_lock_init(&batch_exited[i]);
		spin_lock_init(&batch_lock[i]);
	}
}

static void record_batch(int dev) {
	u64 f2;
	window_size_hist[waiting[dev]] += 1;
	waiting[dev] = 0;
	//is_batch_blocked = 0;
}

static int gpu_get_prediction(int idx, int dev) {
	return multi_gpu_outputs[dev][idx*32]>=(multi_gpu_outputs[dev][(idx+1)*32]) ?  false: true;
}

//hack: weights are actually device pointers here
void gpu_predict_batch(char *__feat_vec, int n_vecs, long **weights) {
	//do inference
	void *args[] = {
		&weights[0], &weights[2], &d_input_vec_i, &d_mid_res_i
	};
	void *args1[] = {
		&weights[1], &weights[3], &d_mid_res_i, &d_final_res_i
	};

    check_error(cuLaunchKernel(batch_linnos_mid_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

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

//hack: weights are actually device pointers here
void multi_gpu_predict_batch(char *__feat_vec, int n_vecs, long **weights, int dev) {
	//do inference
	void *args[] = {
		&weights[0], &weights[2], &multi_d_input_vec_i[dev], &multi_d_mid_res_i[dev]
	};
	void *args1[] = {
		&weights[1], &weights[3], &multi_d_mid_res_i[dev], &multi_d_final_res_i[dev]
	};

    check_error(cuLaunchKernel(batch_linnos_mid_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

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

void do_gpu_inference(int n_vecs, long **weights, int dev) {
	//u64 s,t;	
	//pr_warn(" ###############  copying %d inputs\n", n_vecs);
	//s = ktime_get_ns();
	multi_copy_inputs_to_gpu(n_vecs, dev);
	multi_gpu_predict_batch(0, n_vecs, weights, dev);
	multi_copy_results_from_gpu(n_vecs, dev);
	//t = ktime_get_ns();
	//pr_warn(" ###############  inference took %llu us\n", (t-s)/1000);
}

//this is what an IO calls when it calls predict()
//SYNCHRONIZATION AND EDGE CASE HELL
bool gpu_batch_entry(char *feat_vec, int n_vecs, long **weights) {
	u64 my_id;
	bool my_prediction, still_wait = false;
	u64 my_arrival, wait_multiplier = 1;
	u32 err, i, this_dev=0;
	unsigned long irqflags, f2;

	//aarrgh i hate hard coding, this better work
	for(i = 0; i < NUMBER_DEVICES ; i++) {
		if(first_weight_ptr_to_dev[i] == weights[0]) {
			this_dev = i;
			break;
		}
	}
	pr_warn("my device is %u\n", this_dev);
	if(i==NUMBER_DEVICES) {
		pr_warn("########### didnt find my device\n");
		return false;
	}

	//pr_warn("pending in %d\n", pending++);
	spin_lock_irqsave(&batch_lock[this_dev], irqflags);
	//this avoids ppl from arriving between a batch is full and the nexts start
	while (is_batch_blocked[this_dev]) {
		//pr_warn("is_batch_blocked, waiting...\n");
		spin_unlock_irqrestore(&batch_lock[this_dev], irqflags);
		wait_for_completion(&batch_block[this_dev]); //whoever is blocking *will* release us
		//pr_warn("###### one awake! fighting for lock\n", my_id);
		//when we wake up, fight for the lock
		spin_lock_irqsave(&batch_lock[this_dev], irqflags);
		//pr_warn("###### got the lock\n", my_id);
		//its unlikely we loop. if we do we got left behind..
	}
	//pr_warn("pending out %d\n", pending--);

	my_id = waiting[this_dev]++;
	//pr_warn("id %d ready to work\n", my_id);
	//copy inputs to intermediary buffer
	memcpy((multi_inputs_to_gpu[this_dev]) +(my_id*LEN_INPUT), feat_vec, LEN_INPUT);

	//if this is the first, we start the window time
	if (my_id == 0) {
		my_arrival = ktime_get_ns();
		window_start_ns[this_dev] = my_arrival;

		spin_lock_irqsave(&batch_exited[this_dev], f2);
		n_exited[this_dev] = 0;
		spin_unlock_irqrestore(&batch_exited[this_dev], f2);

		//pr_warn("id 0 going to sleep!\n");
		//let other ppl enter this batch
		spin_unlock_irqrestore(&batch_lock[this_dev], irqflags);
		//when we wake up from this, we finalize a batch
		err = wait_for_completion_timeout(&finalize_batch[this_dev], usecs_to_jiffies(window_size_ns/1000));
		
		//pr_warn("first is awake!\n");
		//grab lock to stop ppl from entering
		spin_lock_irqsave(&batch_lock[this_dev], irqflags); //maybe RC: someone preempts right before memcpy
		reinit_completion(&finalize_batch[this_dev]); //no one will call complete bc we hold the lock

		//there are three conditions for ppl to get in a batch: lock, is_batch_blocked and batch_block
		is_batch_blocked[this_dev] = 1; //this will not let anyone in, even if we release the lock
		reinit_completion(&batch_block[this_dev]); //this causes blocking

		//bookkeeping
		this_batch_size[this_dev] = waiting[this_dev];
		record_batch(this_dev);
		waiting[this_dev] = 0;
		//pr_warn("batch formed with size %d\n", this_batch_size);
		spin_unlock_irqrestore(&batch_lock[this_dev], irqflags);

		//use cpu
		if(waiting[this_dev] < cpu_gpu_threshold) {
		//if (1) {
			use_cpu_instead[this_dev] = 1;
			my_prediction = false;
			//my_prediction = cpu_prediction_model(feat_vec, n_vecs, weights);
		}
		//use GPU
		else {
			n_used_gpu++;
			use_cpu_instead[this_dev] = 0;

			//let the lock go. if we do a sync command with it, the kernel deadlocks :)
			//incoming reqs will not acquire lock since they will wait on batch_block
			//TODO
			//do_gpu_inference(waiting, gpu_weights[this_dev_idx].weights); 
		 	for (err=0 ; err<128 ; err++)
				multi_gpu_outputs[this_dev][err] = false;
			
			my_prediction = gpu_get_prediction(my_id, this_dev);
		}

		//perhaps this was a timeout and we're the only one, so skip waiting
		if (this_batch_size[this_dev] > 1) {
			//we have completed
			n_exited[this_dev]++;
			//let waiters go
			complete_all(&batch_completed[this_dev]);

			//we wait until everyone leaves
			//pr_warn("waiting for everyone to leave\n");
			wait_for_completion(&finalize_batch[this_dev]);
			reinit_completion(&finalize_batch[this_dev]);
			//pr_warn("everyone left\n");
		}

		is_batch_blocked[this_dev] = 0; //no one will loop
		reinit_completion(&batch_completed[this_dev]);
		spin_lock_irqsave(&batch_lock[this_dev], irqflags);
		
		complete_all(&batch_block[this_dev]);
		spin_unlock_irqrestore(&batch_lock[this_dev], irqflags); //next batch starts when we release this lock
		
		return my_prediction;
	}
	/*
	 *   if we are not the first
	 */
	else {
		my_arrival = ktime_get_ns();
		//check if this batch should be finalized
		if(my_arrival - window_start_ns[this_dev] >= window_size_ns || waiting[this_dev] == max_batch_size) {
			complete(&finalize_batch[this_dev]);
		}
		spin_unlock_irqrestore(&batch_lock[this_dev], irqflags);

		//.. wait until first tell us its done
		wait_for_completion(&batch_completed[this_dev]);
		// err = wait_for_completion_timeout(&batch_completed, usecs_to_jiffies(window_size_ns*200/1000));
		// if(err == 0) {
		// 	pr_warn("############ timeout...\n");
		// }

		spin_lock_irqsave(&batch_exited[this_dev], f2);
		n_exited[this_dev]++;
		//pr_warn("%d have left:  %d/%d\n", n_exited, n_exited, this_batch_size);
		if (n_exited[this_dev] == this_batch_size[this_dev]) {
			complete(&finalize_batch[this_dev]);
		}
		spin_unlock_irqrestore(&batch_exited[this_dev], f2);

		return false;
		//get the result and return
		if (use_cpu_instead[this_dev]) 
			return cpu_prediction_model(feat_vec, n_vecs, weights);
		else 
			return gpu_get_prediction(my_id, this_dev);
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



bool batch_test(char *feat_vec, int n_vecs, long **weights) {
	return false;
}

