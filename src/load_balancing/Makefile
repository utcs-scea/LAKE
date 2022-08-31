obj-m += mllb_kern.o
mllb_kern-objs := consts.o helpers.o main.o

ccflags-y += -I$(src)/../kapi/include -I$(KAVA_ROOT)/include -O3

KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kshm/Module.symvers
KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kernel/Module.symvers

all: uspace
	make -f Makefile.cubin
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
	rm -f utest

uspace:
	nvcc consts.c main.c helpers.c kernels.cu -o utest -lcuda

cubin:
	make -f Makefile.cubin

.PHONY: uspace cubin clean
