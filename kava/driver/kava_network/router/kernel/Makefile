#
# Makefile for the Linux CUDA router
#

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
KAVA_ROOT = $(mkfile_dir)/../../../..

ifneq ($(KERNELRELEASE),)

ccflags-y += -I$(KAVA_ROOT)/klib/cuda -I$(KAVA_ROOT)/include -O0 -g

obj-m += router_kern.o

router_kern-objs := router.o firewall.o packet-collector.o

else

KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	echo Running make with kava root as $(KAVA_ROOT)
	KBUILD_VERBOSE=1 $(MAKE) -C $(KDIR) M=$(shell pwd) modules

clean:
	$(MAKE) -C $(KDIR) M=$(shell pwd) clean

.PHONY : clean

endif
