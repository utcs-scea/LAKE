obj-m += linnos_hook.o
linnos_hook-objs := main.o predictors.o

ccflags-y += -I$(src)/..

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean