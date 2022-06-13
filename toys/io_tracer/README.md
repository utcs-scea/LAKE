# ebpf-io-traces

# BCC dependencies

sudo apt-get -y install bison build-essential cmake flex git libedit-dev \
  libllvm6.0 llvm-6.0-dev libclang-6.0-dev python zlib1g-dev libelf-dev libfl-dev python3-distutils

wget https://github.com/iovisor/bcc/releases/download/v0.24.0/bcc-src-with-submodule.tar.gz

** then follow instructions on bcc github page **

# Clear page cache

echo 1 | sudo tee /proc/sys/vm/drop_caches


# Disable readahead

echo 0 | sudo tee /sys/block/mmcblk0/queue/read_ahead_kb

# Enable the hacked sysctl

sudo sysctl -w vm.hack_force_majfault=0
