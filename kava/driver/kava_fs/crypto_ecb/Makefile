obj-m += kava_ecb.o

kava_ecb-y += main.o \
				aesni/aesni-intel_asm.o \
				aesni/aesni-intel_glue.o \
				aesni/fpu.o
kava_ecb-$(CONFIG_64BIT) += aesni/aesni-intel_avx-x86_64.o \
							aesni/aes_ctrby8_avx-x86_64.o

all:
	$(if $(KAVA_ROOT), \
		rm -f Module.symvers && \
		ln -s $(KAVA_ROOT)/Module.symvers .)
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
