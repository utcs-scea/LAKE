mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
KAVA_ROOT = $(mkfile_dir)/../../..

ifneq ($(KERNELRELEASE),)

ccflags-y += -I$(KAVA_ROOT)/klib/cuda -I$(KAVA_ROOT)/include -O1

obj-m += ksm_kern.o
ksm_kern-objs := ksm.o xxhash.o

else

KDIR ?= /lib/modules/$(shell uname -r)/build

all: ksm.cubin
	$(MAKE) -C $(KDIR) M=$(shell pwd) modules

ksm.cubin:
	make -f Makefile.cubin

clean:
	$(MAKE) -C $(KDIR) M=$(shell pwd) clean

.PHONY : clean

endif
