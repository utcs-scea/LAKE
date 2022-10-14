obj-m += linnos_hook.o
linnos_hook-objs := main.o predictors.o helpers.o variables.o

ccflags-y += -I$(src)/.. -I$(src)/../../kapi/include -O3

KBUILD_EXTRA_SYMBOLS += $(src)/../../kapi/kshm/Module.symvers
KBUILD_EXTRA_SYMBOLS += $(src)/../../kapi/kernel/Module.symvers

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean