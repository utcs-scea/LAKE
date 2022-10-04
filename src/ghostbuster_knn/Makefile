obj-m += ghost_buster.o
ghost_buster-objs := ghostbuster.o

ccflags-y += -I$(src)/../kapi/include -I$(KAVA_ROOT)/include -O3 -march=native -mhard-float -msse

KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kshm/Module.symvers
KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kernel/Module.symvers


all:
	make -f Makefile_cubin
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
	rm -f utest

uspace:
	nvcc weights.c main.c helpers.c kernels.cu -o utest -lcuda

cubin:
	make -f Makefile_cubin

.PHONY: uspace cubin clean