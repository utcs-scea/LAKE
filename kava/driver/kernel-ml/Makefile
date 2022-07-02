mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
KAVA_ROOT = $(mkfile_dir)/../..

ifneq ($(KERNELRELEASE),)

ccflags-y += -I$(KAVA_ROOT)/klib/cuda -I$(KAVA_ROOT)/include -O1 -march=native -mhard-float -msse -w

obj-m += kml_kern.o
kml_kern-objs := driver.o helpers.o weights.o

else

KDIR ?= /lib/modules/$(shell uname -r)/build

all: kml.cubin
	$(MAKE) -C $(KDIR) M=$(shell pwd) modules

kml.cubin:
	make -f Makefile.cubin

utest:
	nvcc driver.c helpers.c kernels.cu weights.c -o utest -lcuda

clean:
	$(MAKE) -C $(KDIR) M=$(shell pwd) clean

.PHONY : clean

endif