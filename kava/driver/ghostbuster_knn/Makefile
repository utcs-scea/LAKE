mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
KAVA_ROOT = $(mkfile_dir)/../..

ifneq ($(KERNELRELEASE),)

ccflags-y += -I$(KAVA_ROOT)/klib/cuda -I$(KAVA_ROOT)/include -O3

obj-m += ghost_buster.o
ghost_buster-objs := ghostbuster.o

else

KDIR ?= /lib/modules/$(shell uname -r)/build

all: knncuda.cubin
	$(MAKE) -C $(KDIR) M=$(shell pwd) modules

knncuda.cubin:
	$(MAKE) -C cuda
	mv cuda/knncuda.cubin .

clean:
	$(MAKE) -C $(KDIR) M=$$PWD clean
	rm -f knncuda.cubin

.PHONY : clean

endif
