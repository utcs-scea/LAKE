Kernel Genann
=============

Run benchmarks
--------------

1. Setup CPU genann library:
    ```shell
    ./user/setup.sh
    ```

    The command will build two versions of libgenann.so: one is CPU version (`user/genann/cpu/libgenann.so`)
    and the other one is backed by CUDA (`user/genann/gpu/build/libgenann.so`).

3. Load kgenann module:
    ```shell
    cd $KAVA_ROOT
    ./scripts/load_genann.sh
    ```
    Run `./scripts/load_genann.sh G=1` if the GPU version is wanted.

3. Compile Genann application modules:
    ```shell
    ./kernel/setup.sh
    ```

4. Load Genann application module. For example, the MNIST example can be loaded by
    ```shell
    ./kernel/load.sh mnist [ex1|ex2|ex3|ex4|mnist]
    ```
    The script also unload the module automatically.

5. Unload modules:
    ```shell
    sudo rmmod mnist_kern  # or ./kernel/unload.sh
    ./$KAVA_ROOT/scripts/unload_all.sh
    ```

Measure utilization
-------------------

1. User-space CPU utilization.
    In `kava-benchmarks/scripts`, run
    ````
    python3 fetch_cpu_stat.py -n runmnist -g -o cpu_stats -p _genann
    ```

2. KAvA CPU and GPU utilization.
    Run
    ````
    python3 fetch_cpu_stat.py -n insmod worker kavad -g -o cpu_stats -p _genann
    ```

3. Parse stats.
    ```
    python3 post_process_cpu_stat.py -i cpu_stats_genann.txt -o cpu_utils_genann.csv
    ```
