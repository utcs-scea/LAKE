OUTPUT := .output
CLANG ?= clang
LLVM_STRIP ?= llvm-strip
BPFTOOL = bpftool
ARCH := $(shell uname -m | sed 's/x86_64/x86/' | sed 's/aarch64/arm64/' | sed 's/ppc64le/powerpc/' | sed 's/mips.*/mips/')
#INCLUDES := -I$(OUTPUT) -I../../libbpf/include/uapi -I$(dir $(VMLINUX))
INCLUDES := -I.
CFLAGS := -g -Wall
ALL_LDFLAGS := $(LDFLAGS) $(EXTRA_LDFLAGS)

CLANG_BPF_SYS_INCLUDES = $(shell $(CLANG) -v -E - </dev/null 2>&1 \
	| sed -n '/<...> search starts here:/,/End of search list./{ s| \(/.*\)|-idirafter \1|p }')

all: kprobe

vmlinux.h:
	bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h

kprobe.bpf.o: kprobe.bpf.c vmlinux.h
	$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH) $(INCLUDES) $(CLANG_BPF_SYS_INCLUDES) -c $(filter %.c,$^) -o $@
	$(Q)$(LLVM_STRIP) -g $@

kprobe.skel.h: kprobe.bpf.o
	$(BPFTOOL) gen skeleton $< > $@
 
kprobe: kprobe.skel.h kprobe.c /home/hfingler/linux-6.0/tools/bpf/bpftool/libbpf/libbpf.a
	$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -lelf -lz -o $@

clean:
	rm -f kprobe.skel.h kprobe kprobe.bpf.o vmlinux.h

.PHONY: all clean