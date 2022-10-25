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

bool NEVER_REJECT = false;

#define _us 1000
//batch variables
const u64 window_size_ns = 500*_us;
const u32 max_batch_size = 9; //this cannot be more than 256 (allocated in main.c)
const u32 cpu_gpu_threshold = 8; //less than this we use cpu
const u64 inter_arrival_threshold = 400*_us;

//use normal (0), +1 or +2
extern u8 model_size;

//batch test variables
u32* window_size_hist; //allocated in main.c of kernel_hook, 128 elements
u32 n_used_gpu = 0;
u32 ios_on_device[NUMBER_DEVICES];

u16 current_batch[NUMBER_DEVICES];
spinlock_t batch_entry[NUMBER_DEVICES];
//GPU inference variables
struct GPU_weights gpu_weights[NUMBER_DEVICES]; //per-ssd weights, we are not going to have more than NUMBER_DEVICES ssds..

spinlock_t per_batch_lock[NUMBER_DEVICES][MAX_DEV_BATCHES];
struct completion batch_completed[NUMBER_DEVICES][MAX_DEV_BATCHES];
struct completion finalize_batch[NUMBER_DEVICES][MAX_DEV_BATCHES];
u16 n_exited[NUMBER_DEVICES][MAX_DEV_BATCHES];
u16 this_batch_size[NUMBER_DEVICES][MAX_DEV_BATCHES];
u16 waiting[NUMBER_DEVICES][MAX_DEV_BATCHES];
u64 window_start_ns[NUMBER_DEVICES][MAX_DEV_BATCHES];
u64 last_arrival[NUMBER_DEVICES][MAX_DEV_BATCHES];
bool use_cpu_instead[NUMBER_DEVICES][MAX_DEV_BATCHES];
//0=idle, 1=id0 waiting, 2=running
bool batch_running[NUMBER_DEVICES][MAX_DEV_BATCHES];

void predictors_mgpu_init(void) {
	int i, j;
	for (i=0 ; i < NUMBER_DEVICES ; i++) {
		current_batch[i] = 0;
		ios_on_device[i] = 0;
		spin_lock_init(&batch_entry[i]);
		for (j=0 ; j < MAX_DEV_BATCHES ; j++) {
			n_exited[i][j] = 0;
			this_batch_size[i][j] = 0;
			window_start_ns[i][j] = 0;
			waiting[i][j] = 0;
			batch_running[i][j] = false;
			init_completion(&batch_completed[i][j]);
			init_completion(&finalize_batch[i][j]);
			spin_lock_init(&per_batch_lock[i][j]);
		}
	}
}

static int gpu_get_prediction(int dev, int batch, int id) {
	return multi_gpu_outputs[dev][batch][id*64]>=(multi_gpu_outputs[dev][batch][id*64+32]) ?  false: true;
}

//hack: weights are actually device pointers here
void multi_gpu_predict_batch(char *__feat_vec, int n_vecs, long **weights, int dev, int batch) {
	//do inference
	void *args[] = {
		&weights[0], &weights[2], &multi_d_input_vec_i[dev][batch], &multi_d_mid_res_i[dev][batch]
	};
	void *args1[] = {
		&weights[1], &weights[3], &multi_d_mid_res_i[dev][batch], &multi_d_final_res_i[dev][batch]
	};

    check_error(cuLaunchKernel(batch_linnos_mid_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                cu_streams[dev][batch], 
				args, NULL),
			"cuLaunchKernel", __LINE__);

    check_error(cuLaunchKernel(batch_linnos_final_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				64, 1, 1,   //threads per block
				0,   //shared mem
                cu_streams[dev][batch], 
				args1, NULL),
			"cuLaunchKernel", __LINE__);
}

void multi_gpu_predict_batch_plus_1(char *__feat_vec, int n_vecs, long **weights, int dev, int batch) {
	//do inference
	void *args[] = {
		&weights[0], &weights[2], &multi_d_input_vec_i[dev][batch], &multi_d_mid_res_i[dev][batch]
	};
	void *args1[] = {
		&weights[1], &weights[3], &multi_d_mid_res_1_i[dev][batch], &multi_d_final_res_i[dev][batch]
	};

	void *args2[] = {
		&weights[4], &weights[5], &multi_d_mid_res_i[dev][batch], &multi_d_mid_res_1_i[dev][batch]
	};

    check_error(cuLaunchKernel(batch_linnos_mid_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                cu_streams[dev][batch], 
				args, NULL),
			"cuLaunchKernel", __LINE__);

	check_error(cuLaunchKernel(batch_linnos_mid_layer_1_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args2, NULL),
			"cuLaunchKernel", __LINE__);

    check_error(cuLaunchKernel(batch_linnos_final_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				64, 1, 1,   //threads per block
				0,   //shared mem
                cu_streams[dev][batch], 
				args1, NULL),
			"cuLaunchKernel", __LINE__);
}

void multi_gpu_predict_batch_plus_2(char *__feat_vec, int n_vecs, long **weights, int dev, int batch) {
	//do inference
	void *args[] = {
		&weights[0], &weights[2], &multi_d_input_vec_i[dev][batch], &multi_d_mid_res_i[dev][batch]
	};
	void *args1[] = {
		&weights[1], &weights[3], &multi_d_mid_res_2_i[dev][batch], &multi_d_final_res_i[dev][batch]
	};

	void *args2[] = {
		&weights[4], &weights[5], &multi_d_mid_res_i[dev][batch], &multi_d_mid_res_1_i[dev][batch]
	};

	void *args3[] = {
		&weights[6], &weights[7], &multi_d_mid_res_1_i[dev][batch], &multi_d_mid_res_2_i[dev][batch]
	};

    check_error(cuLaunchKernel(batch_linnos_mid_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                cu_streams[dev][batch], 
				args, NULL),
			"cuLaunchKernel", __LINE__);

	check_error(cuLaunchKernel(batch_linnos_mid_layer_1_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args2, NULL),
			"cuLaunchKernel", __LINE__);

	check_error(cuLaunchKernel(batch_linnos_mid_layer_2_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args3, NULL),
			"cuLaunchKernel", __LINE__);

    check_error(cuLaunchKernel(batch_linnos_final_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				64, 1, 1,   //threads per block
				0,   //shared mem
                cu_streams[dev][batch], 
				args1, NULL),
			"cuLaunchKernel", __LINE__);
}

void do_gpu_inference(int n_vecs, long **weights, int dev, int batch_id) {
	multi_copy_inputs_to_gpu(n_vecs, dev, batch_id);
	multi_gpu_predict_batch(0, n_vecs, weights, dev, batch_id);
	//multi_gpu_predict_batch_plus_1(0, n_vecs, weights, dev, batch_id);
	multi_copy_results_from_gpu(n_vecs, dev, batch_id);
}

void do_gpu_inference_plus_one(int n_vecs, long **weights, int dev, int batch_id) {
	multi_copy_inputs_to_gpu(n_vecs, dev, batch_id);
	multi_gpu_predict_batch_plus_1(0, n_vecs, weights, dev, batch_id);
	multi_copy_results_from_gpu(n_vecs, dev, batch_id);
}

void do_gpu_inference_plus_two(int n_vecs, long **weights, int dev, int batch_id) {
	multi_copy_inputs_to_gpu(n_vecs, dev, batch_id);
	multi_gpu_predict_batch_plus_2(0, n_vecs, weights, dev, batch_id);
	multi_copy_results_from_gpu(n_vecs, dev, batch_id);
}

//this is what an IO calls when it calls predict()
bool gpu_batch_entry(char *feat_vec, int n_vecs, long **weights) {
	u16 my_id;
	u16 my_batch;
	bool my_prediction;
	u64 my_arrival, tdiff;
	u32 i, this_dev=99;
	unsigned long irqflags;
	bool use_cpu;

	for(i = 0; i < NUMBER_DEVICES ; i++) {
		if(first_weight_ptr_to_dev[i] == weights[0]) {
			this_dev = i;
			break;
		}
	}
	if (unlikely(this_dev == 99)) {
		pr_warn("COULD NOT FIND DEV\n");
		return false;
	}

	spin_lock_irqsave(&batch_entry[this_dev], irqflags);
enter_again:
	my_batch = current_batch[this_dev];
	//am I welcome in this batch?
	spin_lock_irqsave(&per_batch_lock[this_dev][my_batch], irqflags);  //TODO this serializes again... maybe just try to get it
	my_id = waiting[this_dev][my_batch];
	if (batch_running[this_dev][my_batch] == true || my_id > max_batch_size) {
		//not welcome
		spin_unlock_irqrestore(&per_batch_lock[this_dev][my_batch], irqflags);
		//move to next batch
		//pr_warn("not welcome in batch %d\n", current_batch[this_dev]);
		current_batch[this_dev] = (current_batch[this_dev]+1) % MAX_DEV_BATCHES;
		goto enter_again; //we loop until we find a batch we can enter
	}
	waiting[this_dev][my_batch] += 1;
	spin_unlock_irqrestore(&batch_entry[this_dev], irqflags);

	//i am welcome, still holding this batch's lock
	if (my_id == 0) { //reinit here to avoid race cond.
		//pr_warn("first of batch %d\n",my_batch);
		reinit_completion(&finalize_batch[this_dev][my_batch]); 
		reinit_completion(&batch_completed[this_dev][my_batch]);
		n_exited[this_dev][my_batch] = 0;
		window_start_ns[this_dev][my_batch] = ktime_get_ns();
		last_arrival[this_dev][my_batch] = window_start_ns[this_dev][my_batch];
	}
	//copy inputs to intermediary buffer, but we need to convert into longs for gpu
	for (i = 0 ; i < LEN_INPUT ; i++)
		multi_inputs_to_gpu[this_dev][my_batch][my_id*LEN_INPUT+i] = (long) feat_vec[i];

	/*
	 * if this is the first, we start the window timer
	 */
	if (my_id == 0) {
		spin_unlock_irqrestore(&per_batch_lock[this_dev][my_batch], irqflags);
		//when we wake up from this, we finalize a batch
		wait_for_completion_timeout(&finalize_batch[this_dev][my_batch], usecs_to_jiffies(window_size_ns/1000));
		
		//when we get this lock, no one is welcome to this batch
		spin_lock_irqsave(&per_batch_lock[this_dev][my_batch], irqflags);
		//this avoids ppl entering even if we release the lock
		batch_running[this_dev][my_batch] = true;
		window_size_hist[waiting[this_dev][my_batch]] += 1;
		//we have to exclude ppl from waking us multiple times, they check based on waiting == 0
		this_batch_size[this_dev][my_batch] = waiting[this_dev][my_batch];
		waiting[this_dev][my_batch] = 0;
		reinit_completion(&finalize_batch[this_dev][my_batch]); 
		spin_unlock_irqrestore(&per_batch_lock[this_dev][my_batch], irqflags);
		
		if(this_batch_size[this_dev][my_batch] == 1) {
			//lonely request :(
			//pr_warn("single request on batch %d\n", my_batch);
			batch_running[this_dev][my_batch] = false;
			my_prediction = cpu_prediction_model(feat_vec, n_vecs, weights);
			return NEVER_REJECT ? false : my_prediction; 
		} else if(this_batch_size[this_dev][my_batch] < cpu_gpu_threshold) {
			use_cpu_instead[this_dev][my_batch] = true;
			complete_all(&batch_completed[this_dev][my_batch]); //XXX
			batch_running[this_dev][my_batch] = false;
			my_prediction = cpu_prediction_model(feat_vec, n_vecs, weights);
			return NEVER_REJECT ? false : my_prediction;
		}
		else {
			n_used_gpu++;
			use_cpu_instead[this_dev][my_batch] = false;

			if (model_size == 0)
				do_gpu_inference(this_batch_size[this_dev][my_batch], gpu_weights[this_dev].weights, this_dev, my_batch); 
			else if (model_size == 1)
				do_gpu_inference_plus_one(this_batch_size[this_dev][my_batch], gpu_weights[this_dev].weights, this_dev, my_batch); 
			else
				do_gpu_inference_plus_two(this_batch_size[this_dev][my_batch], gpu_weights[this_dev].weights, this_dev, my_batch); 
	
			//use GPU
			//for (i=0 ; i<128 ; i++) //fake inference for testin
			//	multi_gpu_outputs[this_dev][my_batch][i] = false;
			my_prediction = gpu_get_prediction(this_dev, my_batch, my_id);
		}
		//we have completed
		n_exited[this_dev][my_batch] += 1;

		//let waiters go
		complete_all(&batch_completed[this_dev][my_batch]);
		//wait for them to exit, reinit
		//pr_warn(" >>>>>:  %d/%d/%d FIRST waiting for non-firsts ...\n", this_dev, my_batch, my_id);
		wait_for_completion(&finalize_batch[this_dev][my_batch]);
		//reinit_completion(&batch_completed[this_dev][my_batch]);
	
		//XXX
		//if (use_cpu_instead[this_dev][my_batch])
		//	my_prediction = cpu_prediction_model(feat_vec, n_vecs, weights);
		//pr_warn(" >>>>>:  %d/%d/%d FIRST WAS LET GO!  batch %d is done\n", this_dev, my_batch, my_id, my_batch);

		batch_running[this_dev][my_batch] = false;
		return NEVER_REJECT ? false : my_prediction;
	}
	/*
	 *   if we are not the first
	 */
	else {
		//spin_lock_irqsave(&per_batch_lock[this_dev][my_batch], irqflags);
		my_arrival = ktime_get_ns();
		tdiff = my_arrival - last_arrival[this_dev][my_batch];
		last_arrival[this_dev][my_batch] = my_arrival;
		//check if this batch should be finalized
		if( waiting[this_dev][my_batch] != 0  &&  //first cannot be awake
				(my_arrival - window_start_ns[this_dev][my_batch] >= window_size_ns //window time
				|| waiting[this_dev][my_batch] == max_batch_size  //batch is full
				|| tdiff >= inter_arrival_threshold) ) {   //too long since someone arrived
			complete(&finalize_batch[this_dev][my_batch]);
		}
		spin_unlock_irqrestore(&per_batch_lock[this_dev][my_batch], irqflags);

		//.. wait until first tell us its done
		//pr_warn("%d/%d/%d: waiting\n", this_dev, my_batch, my_id);
		wait_for_completion(&batch_completed[this_dev][my_batch]);

		use_cpu = use_cpu_instead[this_dev][my_batch];
		if (use_cpu) {
			my_prediction = cpu_prediction_model(feat_vec, n_vecs, weights);
			return NEVER_REJECT ? false : my_prediction;
		}
		
		if (!use_cpu) 
			my_prediction = gpu_get_prediction(this_dev, my_batch, my_id);
		
		spin_lock_irqsave(&per_batch_lock[this_dev][my_batch], irqflags);
		n_exited[this_dev][my_batch] += 1;
		//pr_warn("%d/%d/%d:  %d/%d left\n", this_dev, my_batch, my_id, n_exited[this_dev][my_batch], this_batch_size[this_dev][my_batch]);
		if (n_exited[this_dev][my_batch] == this_batch_size[this_dev][my_batch]) {
			complete(&finalize_batch[this_dev][my_batch]);
			//pr_warn("%d/%d/%d: Waking up first!", this_dev, my_batch, my_id);
		}
		spin_unlock_irqrestore(&per_batch_lock[this_dev][my_batch], irqflags);

		//if its cpu we can tell we exited and do the inference later (here)
		//if (use_cpu)
		//	my_prediction = cpu_prediction_model(feat_vec, n_vecs, weights);
		return NEVER_REJECT ? false : my_prediction;
	}
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

void gpu_predict_batch_plus_1(char *__feat_vec, int n_vecs, long **weights) {
	//do inference
	void *args[] = {
		&weights[0], &weights[2], &d_input_vec_i, &d_mid_res_i
	};
	void *args1[] = {
		&weights[1], &weights[3], &d_mid_res_1_i, &d_final_res_i
	};

	void *args2[] = {
		&weights[4], &weights[5], &d_mid_res_i, &d_mid_res_1_i
	};

    check_error(cuLaunchKernel(batch_linnos_mid_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

	check_error(cuLaunchKernel(batch_linnos_mid_layer_1_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args2, NULL),
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

void gpu_predict_batch_plus_2(char *__feat_vec, int n_vecs, long **weights) {
	//do inference
	void *args[] = {
		&weights[0], &weights[2], &d_input_vec_i, &d_mid_res_i
	};
	void *args1[] = {
		&weights[1], &weights[3], &d_mid_res_2_i, &d_final_res_i
	};

	void *args2[] = {
		&weights[4], &weights[5], &d_mid_res_i, &d_mid_res_1_i
	};

	void *args3[] = {
		&weights[6], &weights[7], &d_mid_res_1_i, &d_mid_res_2_i
	};

    check_error(cuLaunchKernel(batch_linnos_mid_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

	check_error(cuLaunchKernel(batch_linnos_mid_layer_1_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args2, NULL),
			"cuLaunchKernel", __LINE__);

	check_error(cuLaunchKernel(batch_linnos_mid_layer_2_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args3, NULL),
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

bool fake_prediction_model(char *feat_vec, int n_vecs, long **weights) {
	//pr_warn("FAKE\n");
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

    return (final_res_i[0]>=final_res_i[1])? false: true;
}

bool cpu_prediction_model_plus_1(char *feat_vec, int n_vecs, long **weights) {
	long input_vec_i[LEN_INPUT], mid_res_i[LEN_LAYER_0], mid_res_m_1[LEN_LAYER_M_1], final_res_i[LEN_LAYER_1];
	long *weight_0_T_ent, * bias_0_ent, *weight_1_T_ent, * bias_1_ent, *weight_M_1, *bias_M_1; 
	int i, j, k, offset;

	for (i=0 ; i<LEN_INPUT; i++) {
		input_vec_i[i] = (long)(feat_vec[i]);
	}

	weight_0_T_ent = weights[0];
	weight_1_T_ent = weights[1];
	bias_0_ent = weights[2];
	bias_1_ent = weights[3];

	weight_M_1 = weights[4];
	bias_M_1 = weights[5];

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

	for (j = 0; j < LEN_LAYER_M_1; j++) {
		mid_res_m_1[j] = 0;
		for(int off = 0; off < LEN_LAYER_0; off++) {
			mid_res_m_1[j] += mid_res_i[off]*weight_M_1[j * LEN_LAYER_M_1 + off];
		}

		// apply bias
		mid_res_m_1[j] += bias_M_1[j];
		// relu
		if (mid_res_m_1[j] < 0) {
			mid_res_m_1[j] = 0;
		}
	 }
	
	final_res_i[0] = 0;
	for(k=0; k<LEN_LAYER_0; k += 8) {
		final_res_i[0] += (mid_res_m_1[k] == 0 || weight_1_T_ent[k] == 0)? 0 : mid_res_m_1[k] * weight_1_T_ent[k];
		final_res_i[0] += (mid_res_m_1[k+1] == 0 || weight_1_T_ent[k+1] == 0)? 0 : mid_res_m_1[k+1] * weight_1_T_ent[k+1];
		final_res_i[0] += (mid_res_m_1[k+2] == 0 || weight_1_T_ent[k+2] == 0)? 0 : mid_res_m_1[k+2] * weight_1_T_ent[k+2];
		final_res_i[0] += (mid_res_m_1[k+3] == 0 || weight_1_T_ent[k+3] == 0)? 0 : mid_res_m_1[k+3] * weight_1_T_ent[k+3];
		final_res_i[0] += (mid_res_m_1[k+4] == 0 || weight_1_T_ent[k+4] == 0)? 0 : mid_res_m_1[k+4] * weight_1_T_ent[k+4];
		final_res_i[0] += (mid_res_m_1[k+5] == 0 || weight_1_T_ent[k+5] == 0)? 0 : mid_res_m_1[k+5] * weight_1_T_ent[k+5];
		final_res_i[0] += (mid_res_m_1[k+6] == 0 || weight_1_T_ent[k+6] == 0)? 0 : mid_res_m_1[k+6] * weight_1_T_ent[k+6];
		final_res_i[0] += (mid_res_m_1[k+7] == 0 || weight_1_T_ent[k+7] == 0)? 0 : mid_res_m_1[k+7] * weight_1_T_ent[k+7];
	}
	// apply bias
	final_res_i[0] += bias_1_ent[0];

	final_res_i[1] = 0;
	for(k=0; k<LEN_LAYER_0; k += 8) {
		final_res_i[1] += (mid_res_m_1[k] == 0 || weight_1_T_ent[k+256] == 0)? 0 : mid_res_m_1[k] * weight_1_T_ent[k+256];
		final_res_i[1] += (mid_res_m_1[k+1] == 0 || weight_1_T_ent[k+257] == 0)? 0 : mid_res_m_1[k+1] * weight_1_T_ent[k+257];
		final_res_i[1] += (mid_res_m_1[k+2] == 0 || weight_1_T_ent[k+258] == 0)? 0 : mid_res_m_1[k+2] * weight_1_T_ent[k+258];
		final_res_i[1] += (mid_res_m_1[k+3] == 0 || weight_1_T_ent[k+259] == 0)? 0 : mid_res_m_1[k+3] * weight_1_T_ent[k+259];
		final_res_i[1] += (mid_res_m_1[k+4] == 0 || weight_1_T_ent[k+260] == 0)? 0 : mid_res_m_1[k+4] * weight_1_T_ent[k+260];
		final_res_i[1] += (mid_res_m_1[k+5] == 0 || weight_1_T_ent[k+261] == 0)? 0 : mid_res_m_1[k+5] * weight_1_T_ent[k+261];
		final_res_i[1] += (mid_res_m_1[k+6] == 0 || weight_1_T_ent[k+262] == 0)? 0 : mid_res_m_1[k+6] * weight_1_T_ent[k+262];
		final_res_i[1] += (mid_res_m_1[k+7] == 0 || weight_1_T_ent[k+263] == 0)? 0 : mid_res_m_1[k+7] * weight_1_T_ent[k+263];
	}
	// apply bias
	final_res_i[1] += bias_1_ent[1];

    return (final_res_i[0]>=final_res_i[1])? false: true;
}

bool cpu_prediction_model_plus_2(char *feat_vec, int n_vecs, long **weights) {
	long input_vec_i[LEN_INPUT], mid_res_i[LEN_LAYER_0], mid_res_m_1[LEN_LAYER_M_1], mid_res_m_2[LEN_LAYER_M_2], final_res_i[LEN_LAYER_1];
	long *weight_0_T_ent, * bias_0_ent, *weight_1_T_ent, * bias_1_ent, *weight_M_1, *bias_M_1, *weight_M_2, *bias_M_2; 
	int i, j, k, offset;

	for (i=0 ; i<LEN_INPUT; i++) {
		input_vec_i[i] = (long)(feat_vec[i]);
	}

	weight_0_T_ent = weights[0];
	weight_1_T_ent = weights[1];
	bias_0_ent = weights[2];
	bias_1_ent = weights[3];

	weight_M_1 = weights[4];
	bias_M_1 = weights[5];

	weight_M_2 = weights[6];
	bias_M_2 = weights[7];

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

	for (j = 0; j < LEN_LAYER_M_1; j++) {
		mid_res_m_1[j] = 0;
		for(int off = 0; off < LEN_LAYER_0; off++) {
			mid_res_m_1[j] += mid_res_i[off]*weight_M_1[j * LEN_LAYER_M_1 + off];
		}

		// apply bias
		mid_res_m_1[j] += bias_M_1[j];
		// relu
		if (mid_res_m_1[j] < 0) {
			mid_res_m_1[j] = 0;
		}
	 }

	 for (j = 0; j < LEN_LAYER_M_2; j++) {
		mid_res_m_2[j] = 0;
		for(int off = 0; off < LEN_LAYER_M_1; off++) {
			mid_res_m_2[j] += mid_res_m_1[off]*weight_M_2[j * LEN_LAYER_M_2 + off];
		}

		// apply bias
		mid_res_m_2[j] += bias_M_2[j];
		// relu
		if (mid_res_m_2[j] < 0) {
			mid_res_m_2[j] = 0;
		}
	 }
	
	final_res_i[0] = 0;
	for(k=0; k<LEN_LAYER_0; k += 8) {
		final_res_i[0] += (mid_res_m_2[k] == 0 || weight_1_T_ent[k] == 0)? 0 : mid_res_m_2[k] * weight_1_T_ent[k];
		final_res_i[0] += (mid_res_m_2[k+1] == 0 || weight_1_T_ent[k+1] == 0)? 0 : mid_res_m_2[k+1] * weight_1_T_ent[k+1];
		final_res_i[0] += (mid_res_m_2[k+2] == 0 || weight_1_T_ent[k+2] == 0)? 0 : mid_res_m_2[k+2] * weight_1_T_ent[k+2];
		final_res_i[0] += (mid_res_m_2[k+3] == 0 || weight_1_T_ent[k+3] == 0)? 0 : mid_res_m_2[k+3] * weight_1_T_ent[k+3];
		final_res_i[0] += (mid_res_m_2[k+4] == 0 || weight_1_T_ent[k+4] == 0)? 0 : mid_res_m_2[k+4] * weight_1_T_ent[k+4];
		final_res_i[0] += (mid_res_m_2[k+5] == 0 || weight_1_T_ent[k+5] == 0)? 0 : mid_res_m_2[k+5] * weight_1_T_ent[k+5];
		final_res_i[0] += (mid_res_m_2[k+6] == 0 || weight_1_T_ent[k+6] == 0)? 0 : mid_res_m_2[k+6] * weight_1_T_ent[k+6];
		final_res_i[0] += (mid_res_m_2[k+7] == 0 || weight_1_T_ent[k+7] == 0)? 0 : mid_res_m_2[k+7] * weight_1_T_ent[k+7];
	}
	// apply bias
	final_res_i[0] += bias_1_ent[0];

	final_res_i[1] = 0;
	for(k=0; k<LEN_LAYER_0; k += 8) {
		final_res_i[1] += (mid_res_m_2[k] == 0 || weight_1_T_ent[k+256] == 0)? 0 : mid_res_m_2[k] * weight_1_T_ent[k+256];
		final_res_i[1] += (mid_res_m_2[k+1] == 0 || weight_1_T_ent[k+257] == 0)? 0 : mid_res_m_2[k+1] * weight_1_T_ent[k+257];
		final_res_i[1] += (mid_res_m_2[k+2] == 0 || weight_1_T_ent[k+258] == 0)? 0 : mid_res_m_2[k+2] * weight_1_T_ent[k+258];
		final_res_i[1] += (mid_res_m_2[k+3] == 0 || weight_1_T_ent[k+259] == 0)? 0 : mid_res_m_2[k+3] * weight_1_T_ent[k+259];
		final_res_i[1] += (mid_res_m_2[k+4] == 0 || weight_1_T_ent[k+260] == 0)? 0 : mid_res_m_2[k+4] * weight_1_T_ent[k+260];
		final_res_i[1] += (mid_res_m_2[k+5] == 0 || weight_1_T_ent[k+261] == 0)? 0 : mid_res_m_2[k+5] * weight_1_T_ent[k+261];
		final_res_i[1] += (mid_res_m_2[k+6] == 0 || weight_1_T_ent[k+262] == 0)? 0 : mid_res_m_2[k+6] * weight_1_T_ent[k+262];
		final_res_i[1] += (mid_res_m_2[k+7] == 0 || weight_1_T_ent[k+263] == 0)? 0 : mid_res_m_2[k+7] * weight_1_T_ent[k+263];
	}
	// apply bias
	final_res_i[1] += bias_1_ent[1];

    return (final_res_i[0]>=final_res_i[1])? false: true;
}



bool batch_test(char *feat_vec, int n_vecs, long **weights) {
	return false;
}

