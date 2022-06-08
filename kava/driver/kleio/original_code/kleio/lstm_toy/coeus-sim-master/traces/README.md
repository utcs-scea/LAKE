# Memory Access Trace Collection

Methodology to get last-level cache misses.

## Native LLC Miss Rate

Run application with large data input in a native system and observe last level cache miss rate using perf.
```
perf stat -e task-clock,cycles,instructions,cache-references,cache-misses <application>
```
Observe memory footprint in native system.
```
watch -n1 numastat -p <application_name>
```
## Simulation LLC Miss Rate

Install Pin and run the cache simulator to observe the last level cache miss rate. 
```
cd source/tools/Memory/
make obj-intel64/allcache.so
../../../pin -t obj-intel64/allcache.so -- <executable>
```
In order to generate memory access traces of smaller and more manageable size, test:
* Smaller application data inputs.
* Smaller cache size, while preserving the cache size ratio of a native system, e.g. 
```
$ lscpu
L3 (25600kB) + L2 (256KB) + L1d L (32KB) + L1i L (32KB)
```
To change the cache sizes, edit the `allcache.cpp` file and recompile.
Goal is to get a last level cache miss rate in Pintool, similar to the one observed in the native system.

## Intel Pin Trace Collection
* Add the `memtrace.cpp` file from this directory to Intel's Pin `source/tools/Memory/` directory.
* Add `memtrace` to Intel's Pin makefile.rules of the same directory.
* Fix the cache sizes inside the `memtrace.cpp ` file.
```
make obj-intel64/memtrace.so
../../../pin -t obj-intel64/memtrace.so -o <trace.out> -- <executable>
```

## Cori Benchmark Sizes

Intel Pin simulator cache sizes:
```
L3 (2MB) + L2 (32KB) + L1d L (4KB) + L1i L (32KB)
```

Benchmark Data Sizes so that ~600,000 accesses.
```
rodinia_3.1/openmp/backprop/backprop 10000
rodinia_3.1/openmp/kmeans/kmeans_openmp/kmeans -i rodinia_3.1/data/kmeans/inpuGen/5000_34.txt
rodinia_3.1/openmp/lud/omp/lud_omp -s 512
rodinia_3.1/openmp/bfs/bfs 1 rodinia_3.1/data/bfs/inputGen/graph128k.txt
rodinia_3.1/openmp/hotspot/hotspot 256 256 50 1 rodinia_3.1/data/hotspot/temp_256 rodinia_3.1/data/hotspot/power_256 output.out
rodinia_3.1/openmp/b+tree/b+tree.out core 2 file rodinia_3.1/data/b+tree/100k.txt command rodinia_3.1/data/b+tree/command.txt
```
