#
# The following must be provided:
#
KAVA_ROOT ?= /project/kava
UTIL_PATH = $(realpath $(PWD)/../util)

# An executable name
EXECUTABLE	:= heartwall

# C/C++ source files (compiled with gcc)
CCFILES		:= heartwall_cuda.c ./AVI/avilib.c ./AVI/avimod.c

ifneq ($(KERNELRELEASE),)

krn = $(EXECUTABLE)
ccflags-y += -I$(KAVA_ROOT)/include -I$(KAVA_ROOT)/klib/cuda -I$(UTIL_PATH)
obj-m := $(krn).o
$(krn)-objs += $(CCFILES:.c=.o) \
			   ../util/util.o

else

KDIR ?= /lib/modules/$$(uname -r)/build

default:
	$(MAKE) -C $(KDIR) M=$$PWD modules

clean:
	$(MAKE) -C $(KDIR) M=$$PWD clean

.PHONY : clean

endif
