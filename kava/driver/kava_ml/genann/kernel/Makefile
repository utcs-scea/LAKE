# The following must be provided:

TARGET1 = ex1_kern
TARGET2 = ex2_kern
TARGET3 = ex3_kern
TARGET4 = ex4_kern
TARGET5 = mnist_kern
ifneq ($(KERNELRELEASE),)

ccflags-y += -I$(KAVA_ROOT)/include -I$(KAVA_ROOT)/klib/genann \
			 -Wall -Wno-shadow -O3 -g \
			 -march=native -mhard-float -msse

obj-m := $(TARGET1).o $(TARGET2).o $(TARGET3).o $(TARGET4).o $(TARGET5).o
$(TARGET1)-src := example1_kern.c
$(TARGET1)-objs += $($(TARGET1)-src:.c=.o)

$(TARGET2)-src := example2_kern.c
$(TARGET2)-objs += $($(TARGET2)-src:.c=.o)

$(TARGET3)-src := example3_kern.c
$(TARGET3)-objs += $($(TARGET3)-src:.c=.o)

$(TARGET4)-src := example4_kern.c
$(TARGET4)-objs += $($(TARGET4)-src:.c=.o)

$(TARGET5)-src := runmnist.c
$(TARGET5)-objs += $($(TARGET5)-src:.c=.o)


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
