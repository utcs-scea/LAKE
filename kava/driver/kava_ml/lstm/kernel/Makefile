# The following must be provided:

TARGET1 = lstm_kern
ifneq ($(KERNELRELEASE),)

ccflags-y += -I$(KAVA_ROOT)/include -I$(KAVA_ROOT)/klib/lstm_tf \
			 -Wall -Wno-shadow -O3 -g

obj-m := $(TARGET1).o
$(TARGET1)-src := lstm_tf.c
$(TARGET1)-objs += $($(TARGET1)-src:.c=.o)

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
