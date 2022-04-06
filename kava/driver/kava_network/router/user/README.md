CUDA Router
===========

This benchmark implements a subset of a CUDA router, based on
[dtnaylor/cuda-router](https://github.com/dtnaylor/cuda-router).

Instructions
============

You can invoke the router in either CPU-only sequential mode or GPU/CPU mode:

```shell
$ ./router -runtime=3 -cubin firewall.cubin    // GPU/CPU mode
$ ./router -runtime=3 -sequential              // CPU-only mode
```

Without `-packet=n` (number of packets) or `-runtime=n` (seconds of runtime),
the router will keep process packets until being terminated by SIGINT (Ctrl+C).
Other options can be specified at the command line; to see a full list, run:

```shell
$ ./router -help
```

The actual packet processing code is stored in separate .cu files.
This processing code consists of four functions:

```
	process_packets		This is a GPU kernel to perform packet processing
						(like LPM lookup or firewall rule matching). It will
						be called by router.cu every time there is a batch
						of packets to be processed.

	setup_gpu			This function is called only a single time before the
						execution of the process_packets kernel. Any setup
						code needed by the kernel function goes here (e.g., 
						copying forwarding tables or firewall rule sets to
						the GPU).

	process_packets_sequential	This is a CPU-only sequential version of the
						packet processing algorithm, used for comparing
						performance. It is executed whenever there is a new
						batch of packets to process and the router is running
						in CPU-only mode.

	setup_sequential	This function is called only a single time before the
						execution of the process_packets_sequential function.
```

To control what measurements are taken during router execution, use the defines
at the top of router.cu:

	#define MEASURE_LATENCY		// measures min and max packet router latency
	#deifne MEASURE_BANDWIDTH	// measures how many packets per second we process
	#define MEASURE_PROCESSING_MICROBENCHMARK  // measures how long the process
											      function takes to execute
