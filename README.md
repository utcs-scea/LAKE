# Instructions for compiling kernel

## Install BPF and other dependencies

Start with Ubuntu 20 or 22. We assume gcc is installed.
```
sudo apt-get update
sudo apt-get -y install build-essential tmux git pkg-config cmake zsh
sudo apt-get install libncurses-dev gawk flex bison openssl libssl-dev dkms libelf-dev libiberty-dev autoconf zstd
sudo apt-get install libreadline-dev binutils-dev libnl-3-dev
sudo apt-get install libelf-dev libdwarf-dev libdw-dev
git clone https://github.com/acmel/dwarves.git 
cd dwarves/
git checkout tags/v1.22
mkdir build
cd build
cmake -D__LIB=lib ..
sudo make install
sudo /sbin/ldconfig -v
cd ../..
rm -rf dwarves
```

## Compile kernel

Download the linux kernel 
```
wget https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.15.68.tar.xz
tar xf linux-5.15.68.tar.xz
cd linux-5.15.68/
```
Manually copy a config or run `cp /boot/config-$(uname -r) .config`
Copy `linux/set_configs.sh` into linux-5.15.68 dir and run it
Compile with `make -j$(nproc)` (you might have to press enter once)
Then install:
```
sudo make INSTALL_MOD_STRIP=1 modules_install
sudo make install
sudo make headers_install INSTALL_HDR_PATH=/usr
```

Now make the new kernel the default if you are running headless:

Open `/boot/grub/grub.cfg`, write down the id for the advanced menu and the id for the 5.15-hack.
Join them, in that order with a `>`. For example:
`gnulinux-advanced-11b57fec-e05f-4c4d-8d80-445381841fa1>gnulinux-5.15.68-hack-advanced-11b57fec-e05f-4c4d-8d80-445381841fa1`
Open `/etc/default/grub` and at the top add a default option, using the string above. For example:
`GRUB_DEFAULT="gnulinux-advanced-11b57fec-e05f-4c4d-8d80-445381841fa1>gnulinux-5.15.68-hack-advanced-11b57fec-e05f-4c4d-8d80-445381841fa1"`

Since you are here, add to `GRUB_CMDLINE_LINUX_DEFAULT` (create if it doesnt exist):
`cma=128M@0-4G log_buf_len=16M`
For example: `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash cma=128M@0-4G log_buf_len=16M"`
The last argument is optional for more log length

Finally, run `sudo update-grub`.
Reboot and make sure the kernel is right by running `uname -r`

## More BPF stuff

Install llvm and clang.
Add to /etc/apt/sources.list:
```
deb http://archive.ubuntu.com/ubuntu bionic-updates main multiverse restricted universe
deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-15 main
deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic-15 main
```

Clang might not be in path or have its name versioned. If this is true, run the script in scripts/create_llvm_links.sh to rename.
For example, if you have llvm-config-15 and clang-15 but not llvm-config and clang, run:
`create_llvm_links.sh 15 1`


Go to `linux-5.15.65/tools/bpf` (the linux source downloaded in the previous stap), and run `make && sudo make install`

Install libbpf by running:

```
wget https://github.com/libbpf/libbpf/archive/refs/tags/v1.0.0.tar.gz
tar xf v1.0.0.tar.gz
cd libbpf-1.0.0/src
make && sudo make install
cd ../.. ; rm -rf libbpf-1.0.0
```


## Install CUDA

```
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
sudo sh ./cuda_11.7.1_515.65.01_linux.run
```
Select driver and toolkit.
Run `nvidia-smi` to make sure it's working.