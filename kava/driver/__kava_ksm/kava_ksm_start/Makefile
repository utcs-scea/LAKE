obj-m += kava_ksm_start.o

KAVA_ROOT?=/data/$(USER)/kava
ccflags-y += -I$(KAVA_ROOT)/klib/cuda \
             -I$(KAVA_ROOT)/include

all:
	$(if $(KAVA_ROOT), \
		rm -f Module.symvers && \
		ln -s $(KAVA_ROOT)/Module.symvers .)
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
