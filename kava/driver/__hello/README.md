This is a hello world example on how to write and run a driver module using CUDA.

To run, just do `./run.sh`, assuming the `load_all.sh` script is running, which
means `cuda.ko` is loaded and the worker is running.

All debug messages will appear on `dmesg`. You should leave a pane open with `dmesg -w` to 
watch it.


There are many steps involved in writing a new driver. Copy the source files here and change
as required. Makefile might also need to be changed to add more files.

Since we use the CUDA driver API, we need to find the kernel mangled name to manually load
and execute. To do so, run `cuobjdump -all -symbols <your.cubin>` and the symbol will be printed.
For example:

```
symbols:
STT_FUNC         STB_GLOBAL STO_ENTRY      _Z15process_packetsP7_packetPiiiPvi
```

To generate a cubin, your compile command should be something like this:

`nvcc -m64 -O0 --cubin -gencode arch=compute_61,code=sm_61 -I/usr/local/cuda/include -I. -o firewall.cubin firewall.cu -L/usr/local/cuda/lib64 -lcuda`



