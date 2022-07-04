mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
KAVA_ROOT = $(mkfile_dir)/../../..

ifneq ($(KERNELRELEASE),)

ccflags-y += -I$(KAVA_ROOT)/klib/cuda -I$(KAVA_ROOT)/include -O3 

obj-m += kleio_kern.o
kleio_kern-objs := kleio_tf.o 

else

KDIR ?= /lib/modules/$(shell uname -r)/build

all: 
	$(MAKE) -C $(KDIR) M=$(shell pwd) modules

clean:
	$(MAKE) -C $(KDIR) M=$(shell pwd) clean

.PHONY : clean

endif

.PHONY: utest

utest:
	cd ../../../worker/lstm_tf/lstm_tf_wrapper && make
	cp ../../../worker/lstm_tf/lstm_tf_wrapper/c_wrapper.o .
	gcc  kleio_tf.c c_wrapper.o -o utest $(shell python3.6-config --libs)