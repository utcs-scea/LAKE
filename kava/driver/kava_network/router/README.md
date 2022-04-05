CUDA Router
===========

Measure CPU and GPU utilization
-------------------------------

1. CPU utilization.
    In `kava-benchmarks/scripts`, run
    ````
    python3 fetch_cpu_stat.py -n insmod -o cpu_stats -p _net
    ```

2. CPU and GPU utilization.
    Run
    ````
    python3 fetch_cpu_stat.py -n insmod worker kavad -g -o cpu_stats -p _net
    ```

3. Parse stats.
    ```
    python3 post_process_cpu_stat.py -i cpu_stats_net.txt -o cpu_utils_net.csv
    ```
