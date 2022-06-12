mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
KAVA_ROOT = $(mkfile_dir)/../../..

ifneq ($(KERNELRELEASE),)

ccflags-y += -I$(KAVA_ROOT)/klib/cuda -I$(KAVA_ROOT)/include -O1

obj-m += kml_kern.o
kml_kern-objs := driver.o helpers.o weights.o

else

KDIR ?= /lib/modules/$(shell uname -r)/build

all: kml.cubin
	$(MAKE) -C $(KDIR) M=$(shell pwd) modules

kml.cubin:
	make -f Makefile.cubin

clean:
	$(MAKE) -C $(KDIR) M=$(shell pwd) clean

.PHONY : clean

endif
