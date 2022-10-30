obj-m += linnos.o
linnos-objs := variables.o test_weights.o helpers.o main.o predictors.o

ccflags-y += -I$(src)/../kapi/include -I$(KAVA_ROOT)/include -I$(src)/.. -O3  -Wno-declaration-after-statement

KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kshm/Module.symvers
KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kernel/Module.symvers

all:
	make -f Makefile_cubin
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules
	make -f Makefile_cont

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
	rm -f utest

uspace:
	nvcc variables.c weights.c main.c helpers.c kernels.cu predictors.c -o utest -lcuda

.PHONY: uspace cubin clean