#
# Makefile for the Linux eCryptfs
#

obj-m += ecryptfs.o

ecryptfs-objs := dentry.o file.o inode.o main.o super.o mmap.o read_write.o \
	      crypto.o keystore.o kthread.o debug.o

ecryptfs-objs += messaging.o miscdev.o

.PHONY: all

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules
	$(if $(BUILD_DIR), cp ecryptfs.ko $(BUILD_DIR)/ ) 

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
