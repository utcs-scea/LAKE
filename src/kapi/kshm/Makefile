obj-m += lake_shm.o
lake_shm-y += kshm_main.o backend.o
ccflags-y += -I. -I$(src)/../include -O3 -g

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean