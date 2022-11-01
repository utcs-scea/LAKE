obj-m += knn.o
knn-objs := ghostbuster.o

ccflags-y += -I$(src)/../kapi/include -I$(KAVA_ROOT)/include -O3 -march=native -mhard-float -msse

KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kshm/Module.symvers
KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kernel/Module.symvers

all: cubin
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules
	make -C cuda

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
	rm -f utest

cubin: 
	make -f Makefile_cubin

uspace:
	nvcc ghostbuster.c -o utest -lcuda

.PHONY: uspace cubin clean
