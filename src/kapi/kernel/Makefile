obj-$(CONFIG_ECRYPT_FS) += lake_kapi.o

lake_kapi-y := main.o kapi.o netlink.o kargs.c

ccflags-y += -I. -I$(src)/../include -O3

KBUILD_EXTRA_SYMBOLS := $(src)/../kshm/Module.symvers

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean