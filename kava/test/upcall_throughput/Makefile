obj-m += netlink_kern.o spin_shm_kern.o poll_chardev_kern.o ioctl_kern.o

users := $(wildcard *_user.c)
users_prefix := $(users:.c=)

kerns := $(wildcard *_kern.c)
ccflags-y += -O2 -I. -I$(src)/../../include
KDIR ?= /lib/modules/$$(uname -r)/build

all: $(users_prefix) kerns

$(users_prefix): % : %.c
	gcc -O2 $< -I../../include -I. -o $@ common.c -lpthread

kerns: $(kerns)
	$(MAKE) -C $(KDIR) M=$$PWD modules

clean:
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm -f ${users_prefix}

.PHONY: all kerns clean
