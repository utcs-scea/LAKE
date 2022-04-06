mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
KAVA_ROOT = $(mkfile_dir)/../..

ifneq ($(KERNELRELEASE),)

ccflags-y += -I$(KAVA_ROOT)/klib/cuda -I$(KAVA_ROOT)/include -O0 -g

obj-m += hello_kern.o

hello_kern-objs := hello.o

else

KDIR ?= /lib/modules/$(shell uname -r)/build

all: hello.cubin
	KBUILD_VERBOSE=1 $(MAKE) -C $(KDIR) M=$(shell pwd) modules

hello.cubin:
	make -f Makefile.cubin

clean:
	$(MAKE) -C $(KDIR) M=$(shell pwd) clean

.PHONY : clean

endif
