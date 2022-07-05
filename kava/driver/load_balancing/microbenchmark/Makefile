mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
KAVA_ROOT = $(mkfile_dir)/../../..

ifneq ($(KERNELRELEASE),)

ccflags-y += -I$(KAVA_ROOT)/klib/cuda -I$(KAVA_ROOT)/include -O1 -march=native -mhard-float -msse

obj-m += mllb_kern.o
mllb_kern-objs := driver.o helpers.o consts.o

else

KDIR ?= /lib/modules/$(shell uname -r)/build

all: mllb.cubin
	$(MAKE) -C $(KDIR) M=$(shell pwd) modules

mllb.cubin:
	make -f Makefile.cubin

clean:
	$(MAKE) -C $(KDIR) M=$(shell pwd) clean

.PHONY : clean

endif

.PHONY: utest
utest:
	nvcc consts.c driver.c helpers.c kernels.cu -o utest -lcuda
