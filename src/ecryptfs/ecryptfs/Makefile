all:
	make -f Makefile_original
	make -f Makefile_lake

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
