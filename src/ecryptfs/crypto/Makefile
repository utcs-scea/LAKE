
obj-$(CONFIG_ECRYPT_FS) += lake_gcm.o

lake_gcm-y := cuda_gcm.o 

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean