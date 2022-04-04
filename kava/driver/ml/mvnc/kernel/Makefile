# The following must be provided:

TARGET1 = mvnc_hello_world
TARGET2 = mvnc_inception

ifneq ($(KERNELRELEASE),)

ccflags-y += -I$(KAVA_ROOT)/include -I$(KAVA_ROOT)/klib/mvnc \
			 -Wall -Wno-shadow -O3 -g \
			 -march=native -mhard-float -msse \
			 -mpreferred-stack-boundary=4

obj-m := $(TARGET2).o

$(TARGET2)-src := inception.c
$(TARGET2)-objs += $($(TARGET2)-src:.c=.o)

else

$(info $(KAVA_ROOT)/include)
export KROOT=/lib/modules/$(shell uname -r)/build

default: modules

modules:
	$(MAKE) -C $(KROOT) M=$(PWD) modules

modules_install:
	$(MAKE) -C $(KROOT) M=$(PWD) modules_install

kernel_clean:
	$(MAKE) -C $(KROOT) M=$(PWD) clean

clean: kernel_clean
	rm -rf Module.symvers modules.order

.PHONY : clean

endif
