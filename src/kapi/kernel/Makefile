obj-$(CONFIG_ECRYPT_FS) += lake_kapi.o

lake_kapi-y := main.o kapi.o netlink.o 

ccflags-y += -I. -I$(src)/../include -O3

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean