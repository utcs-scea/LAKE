#include <linux/sched/signal.h>
#include <linux/slab.h>
#include <linux/time.h>
#include "router.h"

//#define AVOID_CONTENTION

#define MEASURE_LATENCY
#define MEASURE_BANDWIDTH
//#define MEASURE_MICROBENCHMARKS
//#define MEASURE_CONTENTION

#define DEFAULT_BLOCK_SIZE 32

static bool sequential = 0;
module_param(sequential, bool, 0444);
MODULE_PARM_DESC(sequential, "Run router on CPU, default off");

static int devID = 0;
module_param(devID, int, 0444);
MODULE_PARM_DESC(devID, "GPU device ID in use, default 0");

static char *cubin_path = "firewall.cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to firewall.cubin, default ./firewall.cubin");

static unsigned long max_packet_num = 0;
module_param(max_packet_num, ulong, 0444);
MODULE_PARM_DESC(max_packet_num, "Maximum number of packets to process, default infinity (0)");

static int batch = DEFAULT_BATCH_SIZE;
module_param(batch, int, 0444);
MODULE_PARM_DESC(batch, "The number of packets in a batch, default 1024");

static int wait = DEFAULT_BATCH_WAIT;
module_param(wait, int, 0444);
MODULE_PARM_DESC(wait, "The time we wait (milliseconds) for a complete batch of packets, default 100 ms");

static int block_size = DEFAULT_BLOCK_SIZE;
module_param(block_size, int, 0444);
MODULE_PARM_DESC(block_size, "The number of threads in a block, default 32");

static int runtime = 0;
module_param(runtime, int, 0444);
MODULE_PARM_DESC(runtime, "A runtime in seconds, default 0");

static int input_throughput = DEFAULT_INPUT_THROUGHPUT;
module_param(input_throughput, int, 0444);
MODULE_PARM_DESC(input_throughput, "Throughput of incoming packets, default 0 Gbps (infinite)");

#ifdef FIREWALL
static int numrules = 100;
module_param(numrules, int, 0444);
MODULE_PARM_DESC(numrules, "If processing is firewall, specifies how many rules to generate, default 100");
#endif /* FIREWALL */

struct timespec start_time, cur_time;
bool _do_run = true;
inline bool do_run(void) {
	if (_do_run && runtime > 0) {
		getnstimeofday(&cur_time);

		if (cur_time.tv_sec - start_time.tv_sec > runtime)
			return false;
		else
			return true;
	}
	return _do_run;
}

/**
 * Start a loop:
 *	1) Gather packets
 *  2) Copy packets to GPU and process
 *	3) Copy results back and print performance stats
 *
 * We do this with pipelining: while the GPU is processing one buffer of packets,
 * we're copying over the next batch so that it can begin processing them as soon
 * as it finishes processing the first batch.
 */
#ifdef MEASURE_MICROBENCHMARKS
__attribute__((target("sse")))
#endif /* MEASURE_MICROBENCHMARKS */
static int run_gpu(int block_size)
{
	int i;
    int compute_mode;
    CUdevice dev;
    char dev_name[128];
    int compute_major, compute_minor;
    CUcontext ctx;
    CUmodule mod;
    CUfunction process_packets_func;

	unsigned int buf_size;
	unsigned int results_size;
	packet *h_p1, *h_p2;
	int *h_results1, *h_results2;

	CUdeviceptr d_p1, d_p2;
	CUdeviceptr d_results1, d_results2;

#ifdef AVOID_CONTENTION
  int gpu_util = 0;
  int util_threshold = 80;
  int denominator = 100;
  int numerator = 100;
  int modulus = 0;
  int batch_cnt = 0;
  int next_batch_on_gpu = 1;
#endif /* AVOID_CONTENTION */

#ifdef MEASURE_CONTENTION
  int n_samples = 150000;
  int sample_cnt = 1;
  int batch_size = get_batch_size();
  int batches_per_sample = 5; // TODO: tune this when we figure out GPU perf
  // This is actually the number of packets processed
  u64 *bytes_processed = (u64 *) kmalloc(n_samples * sizeof(u64), GFP_KERNEL);
  struct timespec *batch_times = (struct timespec *) kmalloc(n_samples * sizeof(struct timespec), GFP_KERNEL);
  int idx = 0;
  long time;
#ifndef AVOID_CONTENTION
  int batch_cnt = 0;
#endif /* !AVOID_CONTENTION */
#endif /* MEASURE_CONTENTION */

#ifdef PINNED_PACKET_MEMORY
	unsigned int pflags;
#endif /* PINNED_PACKET_MEMORY */

#ifdef PINNED_MEMORY
	unsigned int flags;
#endif /* PINNED_MEMORY */

#ifdef MEASURE_MICROBENCHMARKS
	// Allocate CUDA events that we'll use for timing
	CUevent start, stop;

	// Allocate regular timevals
	struct timespec micro_get_start, micro_get_stop, micro_send_start, micro_send_stop, micro_copy_to_start, micro_copy_to_stop, micro_copy_from_start, micro_copy_from_stop;

	long avg_micro_proc = 0;
	long avg_micro_get = 0;
	long avg_micro_send = 0;
	long avg_micro_copy_to = 0;
	long avg_micro_copy_from = 0;
	int micro_nIters_proc = 0;
	int micro_nIters_get = 0;
	int micro_nIters_send = 0;
	int micro_nIters_copy_to = 0;
	int micro_nIters_copy_from = 0;
#endif /* MEASURE_MICROBENCHMARKS */

#ifdef MEASURE_BANDWIDTH
	unsigned long long packets_processed = 0;
	struct timespec bw_start, bw_stop;
#endif /* MEASURE_BANDWIDTH */

#ifdef MEASURE_LATENCY
	long max_latency, min_latency, avg_max_latency = 0, avg_min_latency = 0;
	int lat_nIters = 0;
	struct timespec lat_start_oldest1, lat_start_oldest2, lat_start_newest1, lat_start_newest2, lat_stop;
	struct timespec *lat_start_oldest_current;
	struct timespec *lat_start_oldest_next;
	struct timespec *lat_start_newest_current;
	struct timespec *lat_start_newest_next;
#endif /* MEASURE_LATENCY */

	bool data_ready = false;
	bool results_ready = false;
	int num_packets_current;
	int num_packets_next;
	packet *h_p_current;
	packet *h_p_next;
	CUdeviceptr d_p_current;
	CUdeviceptr d_p_next;
	int *h_results_current;
	int *h_results_previous;
	CUdeviceptr d_results_current;
	CUdeviceptr d_results_previous;

#ifdef MEASURE_BANDWIDTH
	long total_time = 0;
	long pkts_per_sec;
	long bits_per_sec;
#endif /* MEASURE_BANDWIDTH */

	int threads_in_block;
	int blocks_in_grid;

	PRINT(V_INFO, "Running CPU/GPU code\n");

    cuInit(0);

	// Get the GPU ready
	check_error(cuDeviceGet(&dev, devID), "cuDeviceGet", __LINE__);
	check_error(cuDeviceGetAttribute(&compute_mode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, dev), "cuDeviceGetAttribute", __LINE__);

	if (compute_mode == CU_COMPUTEMODE_PROHIBITED) {
		printk(KERN_ERR "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cuSetDevice().\n");
		return 0;
	}

	check_error(cuDeviceGetName(dev_name, 128, dev), "cuDeviceGetName", __LINE__);
	check_error(cuDeviceGetAttribute(&compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "cuDeviceGetAttribute", __LINE__);
	check_error(cuDeviceGetAttribute(&compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "cuDeviceGetAttribute", __LINE__);
	PRINT(V_INFO, "GPU Device %d: \"%s\" with compute capability %d.%d\n",
            devID, dev_name, compute_major, compute_minor);

	check_error(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate", __LINE__);

    check_error(cuModuleLoad(&mod, cubin_path), "cuModuleLoad", __LINE__);
    check_error(cuModuleGetFunction(&process_packets_func, mod, "_Z15process_packetsP7_packetPiiiPvi"),
            "cuModuleGetFunction", __LINE__);

	getnstimeofday(&start_time);

    buf_size = sizeof(packet)*get_batch_size();
    results_size = sizeof(int)*get_batch_size();
	PRINT(V_INFO, "Packet buffer size = %d bytes\n", buf_size);

	// Allocate host memory for two batches of up to batch_size packets
	// We will alternate between filling and processing these two buffers
	// (at any given time one of the buffers will either be being filled
	// or being processed)
#ifdef PINNED_PACKET_MEMORY
	pflags = CU_MEMHOSTALLOC_DEVICEMAP;
	check_error(cuMemHostAlloc((void**)&h_p1, buf_size, pflags), "hostAlloc h_p1", __LINE__);
	check_error(cuMemHostAlloc((void**)&h_p2, buf_size, pflags), "hostAlloc h_p2", __LINE__);
#else
	h_p1 = (packet *)kava_alloc(buf_size);
	check_malloc(h_p1, "h_p1", __LINE__);
	h_p2 = (packet *)kava_alloc(buf_size);
	check_malloc(h_p2, "h_p2", __LINE__);
#endif /* PINNED_PACKET_MEMORY */
  	for (i = 0; i < get_batch_size(); i++) {
  	    h_p1[i].payload = (char *)kmalloc(BUF_SIZE*sizeof(char), GFP_KERNEL);
	    check_malloc(h_p1[i].payload, "h_p1[i].payload", __LINE__);
  	    h_p2[i].payload = (char *)kmalloc(BUF_SIZE*sizeof(char), GFP_KERNEL);
	    check_malloc(h_p2[i].payload, "h_p2[i].payload", __LINE__);
  	}

	// Allocate host memory for 2 arrays of results
#ifdef PINNED_MEMORY
	flags = CU_MEMHOSTALLOC_DEVICEMAP;
	check_error(cuMemHostAlloc((void**)&h_results1, results_size, flags), "hostAlloc h_results1", __LINE__);
	check_error(cuMemHostAlloc((void**)&h_results2, results_size, flags), "hostAlloc h_results2", __LINE__);
#else
	h_results1 = (int*)kava_alloc(results_size);
	check_malloc(h_results1, "h_results1", __LINE__);
	h_results2 = (int*)kava_alloc(results_size);
	check_malloc(h_results2, "h_results2", __LINE__);
#endif /* PINNED_MEMORY */

	// Allocate device memory for up to batch_size packets
	// TODO: wait and allocate only the amount needed after we receive?
#ifdef PINNED_PACKET_MEMORY
	check_error(cuMemHostGetDevicePointer(&d_p1, (void *)h_p1, 0), "cuMemHostGetDevicePointer", __LINE__);
	check_error(cuMemHostGetDevicePointer(&d_p2, (void *)h_p2, 0), "cuMemHostGetDevicePointer", __LINE__);
#else
	check_error(cuMemAlloc((CUdeviceptr*) &d_p1, buf_size), "cuMemAlloc d_p1", __LINE__);
	check_error(cuMemAlloc((CUdeviceptr*) &d_p2, buf_size), "cuMemAlloc d_p2", __LINE__);
#endif /* PINNED_PACKET_MEMORY */

	// Allocate device memory for results
#ifdef PINNED_MEMORY
	check_error(cuMemHostGetDevicePointer(&d_results1, (void *)h_results1, 0), "cuMemHostGetDevicePointer", __LINE__);
	check_error(cuMemHostGetDevicePointer(&d_results2, (void *)h_results2, 0), "cuMemHostGetDevicePointer", __LINE__);
#else
	check_error(cuMemAlloc(&d_results1, results_size), "cuMemAlloc d_results1", __LINE__);
	check_error(cuMemAlloc(&d_results2, results_size), "cuMemAlloc_results2", __LINE__);
#endif /* PINNED_MEMORY */

	// Setup execution parameters
	threads_in_block = block_size;
	blocks_in_grid = get_batch_size() / threads_in_block;  // FIXME: optimize if we don't have a full batch
	if (get_batch_size() % threads_in_block != 0) {
		blocks_in_grid++;  // need an extra block for the extra threads
	}

	// Run any processing-specific setup code needed
	// (e.g., this might copy the FIB to GPU for LPM)
	setup_gpu();

#ifdef MEASURE_MICROBENCHMARKS
	check_error(cuEventCreate(&start, 0), "Create start event", __LINE__);
	check_error(cuEventCreate(&stop, 0), "Create stop event", __LINE__);
#endif /* MEASURE_MICROBENCHMARKS */

#ifdef MEASURE_BANDWIDTH
	getnstimeofday(&bw_start);
#endif /* MEASURE_BANDWIDTH */

#ifdef MEASURE_LATENCY
	lat_start_oldest_current = &lat_start_oldest1;
	lat_start_oldest_next = &lat_start_oldest2;
	lat_start_newest_current = &lat_start_newest1;
	lat_start_newest_next = &lat_start_newest2;
#endif /* MEASURE_LATENCY */

	h_p_current = h_p1;
	h_p_next = h_p2;
	d_p_current = d_p1;
	d_p_next = d_p2;
	h_results_current = h_results1;
	h_results_previous = h_results2;
	d_results_current = d_results1;
	d_results_previous = d_results2;

#ifdef MEASURE_CONTENTION
  bytes_processed[0] = 0;
  getnstimeofday(&(batch_times[0]));
#endif /* MEASURE_CONTENTION */

	/* The main loop:
		1) Execute the CUDA kernel
		2) While it's executing, copy the results from the last execution back to the host
		3) While it's executing, copy the packets for the next execution to the GPU */

  PRINT(V_INFO, "packet size: %ld\n", sizeof(packet));
	while(!fatal_signal_pending(current) && do_run()) {
#ifdef MEASURE_CONTENTION
  if (batch_cnt > 0 && batch_cnt % batches_per_sample == 0) {
    if (sample_cnt >= n_samples) {
      PRINT(V_INFO, "vvvvv NEED MORE SAMPLES [%d] vvvvv \n", batch_cnt);
    } else {
      bytes_processed[sample_cnt] = (u64) batch_size * sample_cnt * batches_per_sample * 64;
      getnstimeofday(&(batch_times[sample_cnt]));
      sample_cnt++;
    }
  }
#endif /* MEASURE_CONTENTION */

#if defined(AVOID_CONTENTION) || defined(MEASURE_CONTENTION)
  batch_cnt++;
#endif /* MEASURE_CONTENTION || AVOID_CONTENTION */

		/*************************************************************
		 *                1) EXECUTE THE CUDA KERNEL                 *
		 *************************************************************/
		if (data_ready) { // First execution of loop: data_ready = false
            void *args[] = {
                &d_p_current, &d_results_current, &num_packets_current, &block_size,
                &d_rules, &h_num_rules
            };

#ifdef MEASURE_MICROBENCHMARKS
			// Record the start event
			check_error(cuEventRecord(start, NULL), "Record start event", __LINE__);
#endif /* MEASURE_MICROBENCHMARKS */

#ifdef AVOID_CONTENTION
      // Try to process a batch on the GPU. 
      if (next_batch_on_gpu) { 
        next_batch_on_gpu = 0;
#endif
			// Execute the kernel
			PRINT(V_DEBUG, "vvvvv   Begin processing %d packets   vvvvv\n\n", num_packets_current);
			check_error(cuLaunchKernel(process_packets_func, blocks_in_grid, 1, 1,
                        threads_in_block, 1, 1, 0, NULL, args, NULL),
                    "cuLaunchKernel", __LINE__);

#ifdef AVOID_CONTENTION
      } else {
        // Fall back to CPU
		    process_packets_sequential(h_p_current, h_results_current, num_packets_current);
        PRINT(V_DEBUG, "vvvvv Falling back to CPU vvvvv\n");
      }

      modulus = numerator / denominator;
      PRINT(V_DEBUG, "modulus=%d numerator=%d denominator=%d batch_cnt=%d\n", modulus, numerator, denominator, batch_cnt+ 1);
      // If batch_cntc % modulus == 0, run the next batch on the GPU
      // Otherwise, run it on the CPU
      if (modulus > 0 && batch_cnt % modulus == 0) {
        next_batch_on_gpu = 1;
      }

      // TODO: doing this on every kernel invocation is probably too aggressive, causes some slow-down
      gpu_util = cudaGetGPUUtilizationRates();

      // If utilization is high, multiplicatively back off. Otherwise,
      // additively become more aggressive
      if (gpu_util > util_threshold) {
        numerator /= 2;
      } else if (numerator < denominator) {
        numerator += 2;

        // Make sure we don't go >1
        if (numerator > denominator) {
          numerator = denominator;
        }
      }
#endif /* AVOID_CONTENTION */

#ifdef MEASURE_MICROBENCHMARKS
			// Record the stop event
			check_error(cuEventRecord(stop, NULL), "Record stop event", __LINE__);
#endif /*MEASURE_MICROBENCHMARKS*/


#ifdef MEASURE_BANDWIDTH
			packets_processed += num_packets_current;
#endif /* MEASURE_BANDWIDTH */
		}

		/*************************************************************
		 *          2) COPY BACK RESULTS FROM LAST BATCH             *
		 *************************************************************/
		if (results_ready) { // First and second executions of loop: results_ready = false
			// TODO: double-check that stuff is really executing when I think it is.
			// I think that calling cuEventRecord(stop) right before this will record
			// when the kernel stops executing, but won't block until this actually happens.
			// The cuEventSynchronize call below does block until the kernel stops.
			// So, I think anything we do here will execute on the CPU while the GPU executes
			// the kernel call we made above.

			// Copy the last set of results back from the GPU
#ifdef MEASURE_MICROBENCHMARKS
			getnstimeofday(&micro_copy_from_start);
#endif /* MEASURE_MICROBENCHMARKS */

#ifndef PINNED_MEMORY
#ifdef AVOID_CONTENTION
  // Memcpy if the last batch was executed on the CPU
  if (next_batch_on_gpu) {
#endif /* AVOID CONTENTION */
			check_error(cuMemcpyDtoH(h_results_previous, d_results_previous, results_size), "cuMemcpyDtoH (h_results, d_results)", __LINE__);
#ifdef AVOID_CONTENTION
  } else { } // no-op if the last batch was executed on the CPU
#endif /* AVOID CONTENTION */
#endif /* PINNED_MEMORY */

#ifdef MEASURE_MICROBENCHMARKS
			getnstimeofday(&micro_copy_from_stop);
			avg_micro_copy_from += (micro_copy_from_stop.tv_sec - micro_copy_from_start.tv_sec) * 1000000 + (micro_copy_from_stop.tv_nsec - micro_copy_from_start.tv_nsec) / 1000;
			micro_nIters_copy_from++;
#endif /* MEASURE_MICROBENCHMARKS */

			// Forward packets (right now, h_p_next still holds the *previous* batch)
#ifdef MEASURE_MICROBENCHMARKS
			getnstimeofday(&micro_send_start);
#endif /* MEASURE_MICROBENCHMARKS */
			// send_packets(h_p_next, num_packets_next, h_results_previous);
#ifdef MEASURE_MICROBENCHMARKS
			getnstimeofday(&micro_send_stop);
			avg_micro_send += (micro_send_stop.tv_sec - micro_send_start.tv_sec) * 1000000 + (micro_send_stop.tv_nsec - micro_send_start.tv_nsec) / 1000;
			micro_nIters_send++;
#endif /* MEASURE_MICROBENCHMARKS */

#ifdef MEASURE_LATENCY
			getnstimeofday(&lat_stop);
			max_latency = (lat_stop.tv_sec - lat_start_oldest_current->tv_sec) * 1000000 + (lat_stop.tv_nsec - lat_start_oldest_current->tv_nsec) / 1000;
			min_latency = (lat_stop.tv_sec - lat_start_newest_current->tv_sec) * 1000000 + (lat_stop.tv_nsec - lat_start_newest_current->tv_nsec) / 1000;
			PRINT(V_DEBUG_TIMING, "Latencies from last batch: Max: %ld usec   Min: %ld usec\n", max_latency, min_latency);

			lat_nIters++;
			avg_min_latency += min_latency; // keep a cummulative total and divide later
			avg_max_latency += max_latency;
#endif /* MEASURE_LATENCY */

			// Print results
			PRINT(V_DEBUG, "Results from last batch:\n");
			for (i = 0; i < get_batch_size(); i++) {
				PRINT(V_DEBUG, "%d, ", h_results_previous[i]);
			}
			PRINT(V_DEBUG, "\n\n");
		}

#ifdef MEASURE_BANDWIDTH
		if (max_packet_num > 0 && packets_processed >= max_packet_num)
            break;
#endif /* MEASURE_BANDWIDTH */

		/*************************************************************
		 *                 3) COPY NEXT BATCH TO GPU                 *
		 *************************************************************/
		// Get next batch of packets and copy them to the GPU
		// FIXME: We're forcing the results from the current execution to wait
		// until we get the next batch of packets. Is this OK?
#ifdef MEASURE_LATENCY
		// Approx time we received the first packet of the batch
		// (not perfect if the first packet doesn't arrive immediately)
		getnstimeofday(lat_start_oldest_next);
#endif /* MEASURE_LATENCY */

#ifdef MEASURE_MICROBENCHMARKS
		getnstimeofday(&micro_get_start);
#endif /* MEASURE_MICROBENCHMARKS */
		num_packets_next = 0;
		while (num_packets_next == 0 && !fatal_signal_pending(current) && do_run()) {
			num_packets_next = get_packets(h_p_next);
		}
#ifdef MEASURE_MICROBENCHMARKS
		getnstimeofday(&micro_get_stop);
		avg_micro_get += (micro_get_stop.tv_sec - micro_get_start.tv_sec) * 1000000 + (micro_get_stop.tv_nsec - micro_get_start.tv_nsec) / 1000;
		micro_nIters_get++;
#endif /* MEASURE_MICROBENCHMARKS */

#ifdef MEASURE_LATENCY
		getnstimeofday(lat_start_newest_next);
#endif /* MEASURE_LATENCY */

#ifdef MEASURE_MICROBENCHMARKS
		getnstimeofday(&micro_copy_to_start);
#endif /* MEASURE_MICROBENCHMARKS */

#ifndef PINNED_PACKET_MEMORY
		check_error(cuMemcpyHtoDAsync(d_p_next, h_p_next, buf_size, NULL), "cuMemcpyHtoD (d_p_next, h_p_next)", __LINE__);
#endif /* PINNED_PACKET_MEMORY */

#ifdef MEASURE_MICROBENCHMARKS
		getnstimeofday(&micro_copy_to_stop);
		avg_micro_copy_to += (micro_copy_to_stop.tv_sec - micro_copy_to_start.tv_sec) * 1000000 + (micro_copy_to_stop.tv_nsec - micro_copy_to_start.tv_nsec) / 1000;
		micro_nIters_copy_to++;
#endif /* MEASURE_MICROBENCHMARKS */

		if (data_ready) {
#ifdef MEASURE_MICROBENCHMARKS
			float msecTotal;

			// Wait for the stop event to complete (which waits for the kernel to finish)
			check_error(cuEventSynchronize(stop), "Failed to synchronize stop event", __LINE__);

			check_error(cuEventElapsedTime(&msecTotal, start, stop), "Getting time elapsed b/w events", __LINE__);

			micro_nIters_proc++;
			avg_micro_proc += msecTotal;

			// Compute and print the performance
			PRINT(V_DEBUG_TIMING,
				"Performance= Time= %.3f usec, WorkgroupSize= %u threads/block\n",
				msecTotal,
				threads_in_block);
#else
			// Wait for kernel execution to complete
            // No need to synchronize here as cuMemcpyDtoH will be a sync point.
			// check_error(cuCtxSynchronize(), "cuCtxSynchronize", __LINE__);
#endif /* MEASURE_MICROBENCHMARKS */

			PRINT(V_DEBUG, "^^^^^   Done processing batch   ^^^^^\n\n\n");

			results_ready = true;
		}
		data_ready = true;

		// Get ready for the next loop iteration
		SWAP(num_packets_current, num_packets_next, int);
		SWAP(h_p_current, h_p_next, packet*);
		SWAP(d_p_current, d_p_next, CUdeviceptr);
		SWAP(h_results_current, h_results_previous, int*);
		SWAP(d_results_current, d_results_previous, CUdeviceptr);
#ifdef MEASURE_LATENCY
		SWAP(lat_start_oldest_current, lat_start_oldest_next, struct timespec*);
		SWAP(lat_start_newest_current, lat_start_newest_next, struct timespec*);
#endif /* MEASURE_LATENCY */
	}

#ifdef MEASURE_CONTENTION
  for (idx = 1; i < n_samples; ++idx) {
    // Stop if we had extra room at the end ofhte buffer
    if (idx == sample_cnt) {
      break;
    }
    time = (batch_times[idx].tv_sec - batch_times[0].tv_sec) * 1000000 + (batch_times[idx].tv_nsec - batch_times[0].tv_nsec) / 1000; // us
    PRINT(V_INFO, "CONTENTION:%ld,%lld\n", time, bytes_processed[idx]);
  }
  kfree(bytes_processed);
  kfree(batch_times);
#endif

#ifdef MEASURE_BANDWIDTH
	// Calculate how many packets we processed per second
	getnstimeofday(&bw_stop);
	total_time = (bw_stop.tv_sec - bw_start.tv_sec) * 1000000 + (bw_stop.tv_nsec - bw_start.tv_nsec) / 1000; // us
	PRINT(V_INFO, "Total time = %ld\n", total_time);
	pkts_per_sec = packets_processed * 1000000 / (total_time);
	bits_per_sec = packets_processed * 64 * 8 / (total_time); // 1000 * Gbps

	PRINT(V_INFO, "Bandwidth: %ld packets per second  (64B pkts ==> %ld.%.3ld Gbps)\n",
            pkts_per_sec, bits_per_sec / 1000, bits_per_sec % 1000);
#endif /* MEASURE_BANDWIDTH */

#ifdef MEASURE_MICROBENCHMARKS
	avg_micro_proc /= micro_nIters_proc;
	avg_micro_get /= micro_nIters_get;
	avg_micro_send /= micro_nIters_send;
	avg_micro_copy_to /= micro_nIters_copy_to;
	avg_micro_copy_from /= micro_nIters_copy_from;

	PRINT(V_INFO, "Average processing time: %ld usec\n", avg_micro_proc);
	PRINT(V_INFO, "Average packet get time: %ld usec\n", avg_micro_get);
	PRINT(V_INFO, "Average packet send time: %ld usec\n", avg_micro_send);
	PRINT(V_INFO, "Average packet copy to device time: %ld usec\n", avg_micro_copy_to);
	PRINT(V_INFO, "Average packet copy from device time: %ld usec\n\n", avg_micro_copy_from);
#endif /* MEASURE_MICROBENCHMARKS */

#ifdef MEASURE_LATENCY
	avg_min_latency /= lat_nIters;
	avg_max_latency /= lat_nIters;
	PRINT(V_INFO, "Average latency: Max: %ld usec, Min: %ld usec\n", avg_max_latency, avg_min_latency);
#endif /* MEASURE_LATENCY */

	// Clean up memory
#ifdef PINNED_PACKET_MEMORY
	check_error(cuMemFreeHost(h_p1), "cuMemFreeHost", __LINE__);
	check_error(cuMemFreeHost(h_p2), "cuMemFreeHost", __LINE__);
#else
	kava_free(h_p1);
	kava_free(h_p2);
	cuMemFree(d_p1);
	cuMemFree(d_p2);
#endif /* PINNED_PACKET_MEMORY */
#ifdef PINNED_MEMORY
	check_error(cuMemFreeHost(h_results1), "cuMemFreeHost", __LINE__);
	check_error(cuMemFreeHost(h_results2), "cuMemFreeHost", __LINE__);
#else
	kava_free(h_results1);
	kava_free(h_results2);
	cuMemFree(d_results1);
	cuMemFree(d_results2);
#endif /* PINNED_MEMORY */

#ifdef MEASURE_MICROBENCHMARKS
	check_error(cuEventDestroy(start), "Destroy start event", __LINE__);
	check_error(cuEventDestroy(stop), "Destroy stop event", __LINE__);
#endif /* MEASURE_MICROBENCHMARKS */

	teardown();
	check_error(cuCtxDestroy(ctx), "cuCtxDestroy", __LINE__);

	return 0;
}

static int run_sequential(void)
{
    int i;
	unsigned int buf_size;
	unsigned int results_size;
	packet* p;
	int *results;

#ifdef MEASURE_MICROBENCHMARKS
	struct timespec micro_proc_start, micro_proc_stop, micro_get_start, micro_get_stop, micro_send_start, micro_send_stop;
	long avg_micro_proc = 0;
	long avg_micro_get = 0;
	long avg_micro_send = 0;
	int micro_nIters_proc = 0;
	int micro_nIters_get = 0;
	int micro_nIters_send = 0;
#endif /* MEASURE_MICROBENCHMARKS */

#ifdef MEASURE_BANDWIDTH
	long long packets_processed = 0;
	struct timespec bw_start, bw_stop;
#endif /* MEASURE_BANDWIDTH */

#ifdef MEASURE_LATENCY
	struct timespec lat_start_oldest, lat_start_newest, lat_stop;
	long max_latency, min_latency, avg_max_latency = 0, avg_min_latency = 0;
	int lat_nIters = 0;
#endif /* MEASURE_LATENCY */

#if defined(MEASURE_MICROBENCHMARKS) || defined(MEASURE_BANDWIDTH)
    long total_time;
#endif /*MEASURE_MICROBENCHMARKS*/

#ifdef MEASURE_BANDWIDTH
	long pkts_per_sec;
    long bits_per_sec;
#endif /* MEASURE_BANDWIDTH */

	int num_packets;

	PRINT(V_INFO, "Running sequential router code on CPU only\n");

	getnstimeofday(&start_time);

    buf_size = sizeof(packet)*get_batch_size();
    results_size = sizeof(int)*get_batch_size();

	// Allocate buffer for packets
    p = (packet *)kmalloc(buf_size, GFP_KERNEL);
	check_malloc(p, "p", __LINE__);
  	for(i = 0; i < get_batch_size(); i++) {
  	    p[i].payload = (char *)kmalloc(BUF_SIZE*sizeof(char), GFP_KERNEL);
	    check_malloc(p[i].payload, "p[i].payload", __LINE__);
  	}

	// Allocate array for results
    results = (int*)kmalloc(results_size, GFP_KERNEL);
	check_malloc(results, "results", __LINE__);

	// Run any processing-specific setup code needed
	// (e.g., this might prepare a data structure for LPM)
	setup_sequential();

#ifdef MEASURE_BANDWIDTH
    getnstimeofday(&bw_start);
#endif /* MEASURE_BANDWIDTH */

	/* The main loop:
		1) Get a batch of packets
		2) Process them */
	while(!fatal_signal_pending(current) && do_run()) {
		// Get next batch of packets
#ifdef MEASURE_LATENCY
		getnstimeofday(&lat_start_oldest);
#endif /* MEASURE_LATENCY */

#ifdef MEASURE_MICROBENCHMARKS
		getnstimeofday(&micro_get_start);
#endif /* MEASURE_MICROBENCHMARKS */
		num_packets = 0;
		while (num_packets == 0 && !fatal_signal_pending(current) && do_run()) {
			num_packets = get_packets(p);
		}
#ifdef MEASURE_MICROBENCHMARKS
		getnstimeofday(&micro_get_stop);
		avg_micro_get += (micro_get_stop.tv_sec - micro_get_start.tv_sec) * 1000000 + (micro_get_stop.tv_nsec - micro_get_start.tv_nsec) / 1000;
		micro_nIters_get++;
#endif /* MEASURE_MICROBENCHMARKS */

#ifdef MEASURE_LATENCY
		getnstimeofday(&lat_start_newest);
#endif /* MEASURE_LATENCY */

		// Process the batch
#ifdef MEASURE_MICROBENCHMARKS
		getnstimeofday(&micro_proc_start);
#endif /* MEASURE_MICROBENCHMARKS */

		PRINT(V_DEBUG, "Processing %d packets\n", num_packets);
		process_packets_sequential(p, results, num_packets);

#ifdef MEASURE_MICROBENCHMARKS
		getnstimeofday(&micro_proc_stop);
		total_time = (micro_proc_stop.tv_sec - micro_proc_start.tv_sec) * 1000000 + (micro_proc_stop.tv_nsec - micro_proc_start.tv_nsec) / 1000;

		micro_nIters_proc++;
		avg_micro_proc += total_time;

		PRINT(V_DEBUG_TIMING, "Performance: %ld usec\n", total_time);
#endif /*MEASURE_MICROBENCHMARKS*/

		// Return the batch of packets to click for forwarding
#ifdef MEASURE_MICROBENCHMARKS
		getnstimeofday(&micro_send_start);
#endif /* MEASURE_MICROBENCHMARKS */
		// send_packets(p, num_packets, results);
#ifdef MEASURE_MICROBENCHMARKS
		getnstimeofday(&micro_send_stop);
		avg_micro_send += (micro_send_stop.tv_sec - micro_send_start.tv_sec) * 1000000 + (micro_send_stop.tv_nsec - micro_send_start.tv_nsec) / 1000;
		micro_nIters_send++;
#endif /* MEASURE_MICROBENCHMARKS */

#ifdef MEASURE_LATENCY
		getnstimeofday(&lat_stop);
		max_latency = (lat_stop.tv_sec - lat_start_oldest.tv_sec) * 1000000 + (lat_stop.tv_nsec - lat_start_oldest.tv_nsec) / 1000;
		min_latency = (lat_stop.tv_sec - lat_start_newest.tv_sec) * 1000000 + (lat_stop.tv_nsec - lat_start_newest.tv_nsec) / 1000;
		PRINT(V_DEBUG_TIMING, "Latencies: Max: %ld usec   Min: %ld usec\n", max_latency, min_latency);

		lat_nIters++;
		avg_max_latency += max_latency; // store cummulative latency; divide later
		avg_min_latency += min_latency;
#endif /* MEASURE_LATENCY */

#ifdef MEASURE_BANDWIDTH
		packets_processed += num_packets;
#endif /* MEASURE_BANDWIDTH */

		// Print results
		PRINT(V_DEBUG, "Results:\n");
		for (i = 0; i < get_batch_size(); i++) {
			PRINT(V_DEBUG, "%d, ", results[i]);
		}
		PRINT(V_DEBUG, "\n\n\n");

#ifdef MEASURE_BANDWIDTH
		if (max_packet_num > 0 && packets_processed >= max_packet_num)
            break;
#endif /* MEASURE_BANDWIDTH */
	}

#ifdef MEASURE_BANDWIDTH
	// Calculate how many packets we processed per second
	getnstimeofday(&bw_stop);
	total_time = (bw_stop.tv_sec - bw_start.tv_sec) * 1000000 + (bw_stop.tv_nsec - bw_start.tv_nsec) / 1000; // us
	PRINT(V_INFO, "Total time = %ld\n", total_time);
	pkts_per_sec = packets_processed * 1000000 / (total_time);
	bits_per_sec = packets_processed * 64 * 8 / (total_time); // 1000 * Gbps

	PRINT(V_INFO, "Bandwidth: %ld packets per second  (64B pkts ==> %ld.%.3ld Gbps)\n",
            pkts_per_sec, bits_per_sec / 1000, bits_per_sec % 1000);
#endif /* MEASURE_BANDWIDTH */

#ifdef MEASURE_MICROBENCHMARKS
	avg_micro_proc /= micro_nIters_proc;
	avg_micro_get /= micro_nIters_get;
	avg_micro_send /= micro_nIters_send;

	PRINT(V_INFO, "Average processing time: %ld usec\n", avg_micro_proc);
	PRINT(V_INFO, "Average packet get time: %ld usec\n", avg_micro_get);
	PRINT(V_INFO, "Average packet send time: %ld usec\n\n", avg_micro_send);
#endif /* MEASURE_MICROBENCHMARKS */

#ifdef MEASURE_LATENCY
	avg_max_latency /= lat_nIters;
	avg_min_latency /= lat_nIters;
	PRINT(V_INFO, "Average latency: Max: %ld usec, Min: %ld\n", avg_max_latency, avg_min_latency);
#endif /* MEASURE_LATENCY */

	return 0;
}

/**
 * Program main
 */
static int __init router_init(void)
{
	set_batch_size(batch);
	set_batch_wait(wait);
#ifdef FIREWALL
    set_num_rules(numrules);
#endif /* FIREWALL */
#ifdef SET_INCOMING_THROUGHPUT
    set_input_throughput(input_throughput);
#endif

	// Start the router!
    if (sequential)
		return run_sequential();
	else
		return run_gpu(block_size);

    return 0;
}

static void __exit router_fini(void)
{
    free_packets();
}

module_init(router_init);
module_exit(router_fini);

MODULE_AUTHOR("Hangchen Yu");
MODULE_DESCRIPTION("Kernel module of CUDA router");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
