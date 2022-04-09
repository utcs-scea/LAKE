mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
KAVA_ROOT = $(mkfile_dir)/../../..

ifneq ($(KERNELRELEASE),)

ccflags-y += -I$(KAVA_ROOT)/klib/cuda -I$(KAVA_ROOT)/include -O1

obj-m += linnos_kern.o
linnos_kern-objs := driver.o helpers.o weights.o consts.o

else

KDIR ?= /lib/modules/$(shell uname -r)/build

all: linnos.cubin
	$(MAKE) -C $(KDIR) M=$(shell pwd) modules

linnos.cubin:
	make -f Makefile.cubin

clean:
	$(MAKE) -C $(KDIR) M=$(shell pwd) clean

.PHONY : clean

endif
