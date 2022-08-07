
obj-$(CONFIG_ECRYPT_FS) += lake_gcm.o

lake_gcm-y := gcm_main.o gcm_cuda.o 

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean

.PHONY: test_user
test_user:
	nvcc test_user.c gcm_cuda.c -lcuda -o test_user