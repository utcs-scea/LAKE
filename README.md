






## eBPF

sudo apt-get install libreadline-dev binutils-dev clang-11
Go to linux-5.15.65/tools/bpf, make and make install 

Clang might not be in path, do something like 
sudo ln -s /usr/bin/clang-11 /usr/bin/clang

wget https://github.com/libbpf/libbpf/archive/refs/tags/v1.0.0.tar.gz
untar, cd src, make

sudo mount -t tracefs nodev /sys/kernel/tracing

