mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
KAVA_ROOT = $(mkfile_dir)/../../..

ifneq ($(KERNELRELEASE),)

ccflags-y += -I$(KAVA_ROOT)/klib/cuda -I$(KAVA_ROOT)/include -O0

CFLAGS_mllb_cpu.o := -O0 -mhard-float

obj-m += hack_mllb.o
hack_mllb-objs := driver.o consts.o mllb_cpu.o

else

KDIR ?= /lib/modules/$(shell uname -r)/build

all: 
	$(MAKE) -C $(KDIR) M=$(shell pwd) modules

clean:
	$(MAKE) -C $(KDIR) M=$(shell pwd) clean

.PHONY : clean

endif
