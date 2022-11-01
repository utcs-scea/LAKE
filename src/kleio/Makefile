obj-m += kleio.o
kleio-objs := kleio_tf.o

ccflags-y += -I$(src)/../kapi/include -I$(KAVA_ROOT)/include -O3

KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kshm/Module.symvers
KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kernel/Module.symvers

all: 
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
	rm -f utest

.PHONY: clean