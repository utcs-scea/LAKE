mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
KAVA_ROOT = $(mkfile_dir)/../../..

ifneq ($(KERNELRELEASE),)

ccflags-y += -I$(KAVA_ROOT)/klib/cuda -I$(KAVA_ROOT)/include -O1

obj-m += hackcbc_kern.o
hackcbc_kern-objs := main.o

else

KDIR ?= /lib/modules/$(shell uname -r)/build

all: ecb.cubin
	$(MAKE) -C $(KDIR) M=$(shell pwd) modules

ecb.cubin:
	make -f Makefile_cubin

clean:
	$(MAKE) -C $(KDIR) M=$(shell pwd) clean

.PHONY : clean

endif
