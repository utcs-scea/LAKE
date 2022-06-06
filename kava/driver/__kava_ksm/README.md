# KSM with KAvA-Based CUDA Checksumming

## TODO
* Documentation
* Optimizations
    * Optimize batching
        * Memcpy
        * Single Kernel Launch
    * Other?
* Clean up code
* Benchmark

## Notes
* The scritps in this directory have relative paths. Make sure to modify the scripts/environment file to match your directory structure.
* The Makefile in mm has a relative path to the kava directory. Make sure to adjust this if necessary.
* This is based off the KSM implementation for Linux 4.19.104
* We switched to xxhash because it's faster and used in kernel 5.x.

## Run xxhash without KAvA
Compile CPU libxxhash and then CUDA xxhash test program (similarly, jhash is in `kava_ksm/cu/jhash2`):

```bash
cd kava_ksm/cu/xxhash/cpu
make
cd ../gpu
make
./test_uspace
```

## Installation
1. Make sure you have cloned kava-benchmarks under `$KAVA_ROOT/driver`.
2. Adjust path to the 4.19 kernel installation in `scripts/environment`.
3. Install the modified `kava_ksm`. Change the hash kernel's path `ksm_cubin_path` in `kava_ksm/mm/ksm.c`; change the KAvA root path in `ksm/kava_ksm/mm/Makefile`. Then from the top-level directory of this repository, run:
    ```bash
    ./kava_ksm/scripts/copy_to_kernel.sh
    ./kava_ksm/scripts/build_kernel.sh
    ./kava_ksm/scripts/install_kernel.sh
    ```
4. Build xxhash and `ksm_start` module (which starts `kava_ksm` and registers AvA's CUDA functions).
    ```bash
    ./scripts/build_all.sh
    ```
5. Reboot into the installed kernel. The NVIDIA driver may need to be reinstalled after the reboot.
    ```bash
    sudo ./cuda_10.0.130_410.48_linux --silent --driver --toolkit
    ```

## Run kava\_ksm
1. Start KAvA in a new terminal.
    ```bash
    cd kava
    ./scripts/load_all.sh
    ```

2. Register KAvA's CUDA functions in ksm, set the batch size, and start ksm by running the following script.
    If `batch_size` is zero, KSM falls back to the CPU version.
    ```bash
    ./scripts/start.sh [batch_size]
    ```

3. Measure processor utilization. The scripts are stored in `kava-benchmarks` (cloned into `kava/driver/kava-benchmarks`).
    Run the following scripts to measure the CPU and GPU utilization of KAvA and ksmd:
    ```bash
    mkdir -p result
    python3 fetch_cpu_stat.py -n ksmd worker kavad -gpu -s -d result/ -p _batch_size_256

    # For CPU version, use
    # python3 fetch_cpu_stat.py -n ksmd -gpu -s -d result/ -p _batch_size_0
    ```
    The result can be parsed by
    ```bash
    python3 scripts/post_process_cpu_stat.py -i cpu_stats_batch_size_256.txt -o cpu_stats_batch_size_256.csv
    # or
    python3 scripts/post_process.py -d result
    ```

4. Start samepage generator.
    ```bash
    ./samepage_generator/generator -n 500000

    ```

5. Stop scripts. Stop the generator, fetch script, and KSM (`scripts/stop.sh`) in order. Then stop KAvA.

## Configuration Variables
* sysfs variables
    * `/sys/kernel/mm/ksm/batch_size` : get/set the kava batch size. This can be set once when starting `kava_ksm`.
    * `/sys/kernel/mm/ksm/num_rounds_to_time` : get/set the number of complete scans of the address space to time.
    * `/sys/kernel/mm/ksm/scan_time` : get the time taken to scan all pages `num_rounds_to_time` times.
    * `/sys/kernel/mm/ksm/throughput` : get the aggregate throughput (number of checksummed pages / msec) until now.
    * Read more about the default ksm sysfs variables [here](https://github.com/torvalds/linux/blob/master/Documentation/admin-guide/mm/ksm.rst).
* ksm.c definitions
    * `KAVA_KSM_DEBUG` : print debug messages.
    * `KAVA_KSM_BATCHING` : turn batching on/off.
