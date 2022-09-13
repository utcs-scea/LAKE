import sys
from bcc import BPF
from time import sleep

bpf_text = """
#include <uapi/linux/ptrace.h>

KFUNC_PROBE(hello_test_kfunc_trigger)
{
    bpf_trace_printk("=> from bcc");
    return 0;
}

"""

if BPF.support_kfunc():
	print("BPF does support kfunc :)")
else:
	print("BPF does not support kfunc :(")
	sys.exit(0)

fns = BPF.get_kprobe_functions(b"hello_test_kfunc_trigger")
if fns:
	print("found kprobe for hello_test_kfunc_trigger: ", fns)
else:
	print("kprobe for hello_test_kfunc_trigger NOT FOUND")

b = BPF(text=bpf_text)

print ('probing... ctrl+c to exit')
while True:
    sleep(1)
