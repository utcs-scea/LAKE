# Instructions for compiling kernel

Start with Ubuntu 20 or 22. We assume gcc is installed.

## Install dependencies

```
sudo apt-get update
sudo apt-get -y install build-essential tmux git pkg-config cmake zsh
sudo apt-get install libncurses-dev gawk flex bison openssl libssl-dev dkms libelf-dev libiberty-dev autoconf zstd
sudo apt-get install libreadline-dev binutils-dev libnl-3-dev
sudo apt-get install libelf-dev libdwarf-dev libdw-dev ecryptfs-utils cpufrequtils 
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

Clone `git@github.com:hfingler/linux-6.0.git`.
Go in the directory and run `full_compilation.sh`, it should do everything.

If you are running with a monitor, reboot and choose the new kernel in grub.
Otherwise, make the new kernel the default by:
1. Open `/boot/grub/grub.cfg`, scroll down until you see boot options, then write down the id for the advanced menu and the id for the 6.0-lake.
2. Join them (advanced menu and kernel id), in that order with a `>`. For example:
`gnulinux-advanced-11b57fec-e05f-4c4d-8d80-445381841fa1>gnulinux-6.0-hack-advanced-11b57fec-e05f-4c4d-8d80-445381841fa1`
3. Open `/etc/default/grub` and, at the top of the file, add a default option using the string above. For example:
`GRUB_DEFAULT="gnulinux-advanced-11b57fec-e05f-4c4d-8d80-445381841fa1>gnulinux-5.15.68-hack-advanced-11b57fec-e05f-4c4d-8d80-445381841fa1"`
4. Add to `GRUB_CMDLINE_LINUX_DEFAULT` (create if it doesn't exist): `cma=128M@0-4G log_buf_len=16M`
For example: `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash cma=128M@0-4G log_buf_len=16M"`
5. Finally, run `sudo update-grub`. Reboot and make sure the lake kernel is right by running `uname -r`


## More BPF stuff

Go to `tools/bpf` in the kernel repo you set up above (the `linux-6.0` repo).
and run `make && sudo make install`

Now we need to install llvm and clang.
Add to `/etc/apt/sources.list` (if you're using Ubuntu 22.04, replace `bionic` with `jammy`)
```
deb http://archive.ubuntu.com/ubuntu bionic-updates main multiverse restricted universe
deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-15 main
deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic-15 main
```

Then install it:
```
sudo apt update
sudo apt install llvm-15 clang-15
```

Logout and back in to update path. Try running  `llvm-config --version`.
If it shows `15.0.2`, you are done.
If it does not, make sure `llvm-config-15 --version` works.
This means that the links weren't created, so do it using a script:
go to the `scripts` dir and run `sudo ./create_llvm_links.sh 15 1`.
This will create links to every tool with a `-15` suffix to one without.


## Install CUDA

```
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/515.76/NVIDIA-Linux-x86_64-515.76.run
sudo ./NVIDIA-Linux-x86_64-515.76.run -s
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh ./cuda_11.8.0_520.61.05_linux.run --toolkit --silent --override
```
Run `nvidia-smi` to make sure it's working. If it isn't, the CUDA installer probably uninstalled the driver.
If so, run the second command again (`sudo ./NVIDIA-Linux-x86_64-515.76.run -s`).


# eCryptfs

Mount NVMe
```
sudo parted -a optimal /dev/nvme0n1 mklabel gpt
sudo parted -a optimal /dev/nvme0n1 mkpart primary ext4 0% 100%
sudo mkfs.ext4 /dev/nvme0n1p1
sudo mount /dev/nvme0n1p1 /disk/nvme0 -t ext4
```

Make sure you have python3.8 or above.
If you don't install it by running the command below. We require pip for installing matplotlib, if you already have it, you don't need to install pip.
```
sudo apt install python3.10 python3-pip
```

If the run command gives you `Unable to link the KEY_SPEC_USER_KEYRING into the KEY_SPEC_SESSION_KEYRING`,
it's a bug from using tmux. Close all tmux panes and reopen it with the `tmux.sh` script


#### TODO

plot linnos batch size, color where cpu is used and where gpu is used

# Train the LinnOS Model

## Creating traces

cd LinnOSWriterReplayer
TraceTag='trace'
nohup sudo ./writer /dev/nvme0n1 'testTraces/anonymous.drive0.'$TraceTag &
nohup sudo ./writer /dev/nvme1n1 'testTraces/anonymous.drive1.'$TraceTag &
nohup sudo ./writer /dev/nvme2n1 'testTraces/anonymous.drive2.'$TraceTag &

sudo ./replayer_fail /dev/nvme0n1-/dev/nvme0n1-/dev/nvme2n1 \
 ../testTraces/traindrive0_10xbig.trace \
 ../testTraces/traindrive1_10xbig.trace \
 ../testTraces/traindrive2_10xbig.trace py/TestTraceOutput
 
python3 -m venv linnOSvenv
source linnOSvenv/bin/activate
pip3 install numpy
 
python3 py/percentile.py 2 read \
py/TestTraceOutput py/BaselineData

sudo ./replayer_fail /dev/nvme0n1-/dev/nvme0n1-/dev/nvme2n1 \
 'testTraces/traindrive0.'$TraceTag \
 'testTraces/traindrive1.'$TraceTag \
 'testTraces/traindrive2.'$TraceTag py/TrainTraceOutput
 
## Parse traces and train the model

pip3 install --upgrade pip
pip3 install tensorflow
pip3 install keras
pip3 install pandas
pip3 install scikit-learn

for i in 0 1 2 
do
   python3 py/traceParser.py direct 3 4 \
   py/TrainTraceOutput mlData/temp1 \
   mlData/"mldrive${i}.csv" "$i"
done

for i in 0 1 2 
do
   python3 py/pred1.py \
   mlData/"mldrive${i}.csv" > "mldrive${i}results".txt
done



##stopped here
python3 py/pred1.py mlData/mldrive1.csv > mldrive1results.txt


## Converting the weights to linux header file

cd mlData
mkdir -p drive0weights
mkdir -p drive1weights
mkdir -p drive2weights
cp mldrive0.csv.* drive0weights
cp mldrive1.csv.* drive1weights
cp mldrive2.csv.* drive2weights
 
python3 mlHeaderGen/mlHeaderGen.py Trace nvme0n1 mlData/drive0weights weights_header
python3 mlHeaderGen/mlHeaderGen.py Trace nvme1n1 mlData/drive1weights weights_header
python3 mlHeaderGen/mlHeaderGen.py Trace nvme2n1 mlData/drive2weights weights_header


## Mongodb

To install, go into `src/linnos/mongodb` and run `./install.sh`