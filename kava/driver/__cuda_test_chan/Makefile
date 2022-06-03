krn = test_drv
ccflags-y += -I$(PWD)/../../include -I$(PWD)/../../klib/cuda -O2

ifneq ($(KERNELRELEASE),)

obj-m := $(krn).o
$(krn)-objs += test_chan.o

else

KDIR ?= /lib/modules/$$(uname -r)/build

default:
	$(MAKE) -C $(KDIR) M=$$PWD modules

clean:
	$(MAKE) -C $(KDIR) M=$$PWD clean

.PHONY : clean

endif
