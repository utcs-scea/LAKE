ccflags-y += -I$(src)/../kapi/include -I$(KAVA_ROOT)/include -O0 -g

obj-m += hello_kern.o
hello_kern-objs := hello.o

KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kshm/Module.symvers
KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kernel/Module.symvers

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules
	rm -f hello.cubin
	make -f Makefile_cubin

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean

uspace: uspace.c
	nvcc uspace.c -lcuda -o uspace