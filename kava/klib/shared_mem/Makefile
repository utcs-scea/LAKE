klib_shm = kshm

obj-m += $(klib_shm).o
$(klib_shm)-y += kshm_main.o backend.o test.o
ccflags-y += -I. -I$(src)/../../include -O3 -g
