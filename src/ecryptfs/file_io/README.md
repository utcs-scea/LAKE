File I/O Benchmark
==================

Usage
-----

`./fs_bench <mount point> <repeats> <file size and magnitude> <block size and magnitude>`

The file size must be larger and dividable by the block size.

Example
-------

Run the benchmark with 1 MiB file, 5 repeats, and read/write in 4 KiB blocks to home folder:
`./fs_bench ~/ 5 1M 4K`
