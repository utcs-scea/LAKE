# After the steps below, to launch the VM:

```
virsh domiflist hackvm
arp -an | grep <MAC from previous command>
<find ip>
ssh user@ip_above
```

# Dependencies

`sudo apt install qemu-kvm virtinst libvirt-daemon-system libvirt-clients bridge-utils -y`

# Create image (change directory as needed) and mount

```
virt-install \
--name hackvm \
--ram 2048 \
--disk path=/disk/hfingler/imgs/hackvm.img,size=8 \
--vcpus 1 \
--virt-type kvm \
--os-type linux \
--os-variant ubuntu18.04 \
--graphics none \
--location 'http://archive.ubuntu.com/ubuntu/dists/bionic/main/installer-amd64/' \
--extra-args "console=tty0 console=ttyS0,115200n8"
```

Follow steps on installation ...
Then, to launch the VM:

```
virsh destroy hackvm
virsh start hackvm
virsh domiflist hackvm
arp -an | grep <MAC from previous command>
ssh user@ip_above
```

## Inside the VM, install some stuff we need
```
sudo apt install git emacs tmux zsh
sudo apt install git fakeroot build-essential ncurses-dev xz-utils libssl-dev bc flex libelf-dev bison
```

## If everything works, you can quit either by shutting the VM down from inside or 
`virsh destroy hackvm`


# Compiling linux kernel (do it from INSIDE THE VM for now)

Increase cpu count to make it faster
virsh setvcpus <vm_name> 8 --config --maximum
virsh setvcpus <vm_name> 8 --config

> start the VM, ssh into it
```
wget https://cdn.kernel.org/pub/linux/kernel/v4.x/linux-4.19.237.tar.xz
tar xf linux-4.19.237.tar.xz
cd linux-4.19.237
make defconfig
make kvmconfig
make kvm_guest.config
make olddefconfig
```

Make sure these are in the .config
```
CONFIG_CMA=y
CONFIG_CMA_AREAS=7
CONFIG_DMA_CMA=y
CONFIG_CMA_SIZE_MBYTES=32
CONFIG_CMA_SIZE_SEL_MBYTES=y
CONFIG_CMA_ALIGNMENT=8
CONFIG_INPUT_CMA3000=m
CONFIG_INPUT_CMA3000_I2C=m
```

```
make -j16
sudo make modules_install
sudo make install
```

## If after installation, it doesn't boot:

virsh start hackvm --console

>Choose an older kernel, see if it works.
>No boot log output? Follow https://serverfault.com/a/705062/277700
>No IP assigned, but VM boots? `ip a` to find netif, then `sudo dhclient -v enp0s2`. To make it persistent edit `/etc/netplan/01-netcfg.yaml` and add `enp0s2: dhcp4: yes` (with idents)


# If you need to mount the VM image for any reason

## Mount
```
sudo modprobe nbd max_part=8
sudo qemu-nbd --connect=/dev/nbd0 /disk/hfingler/imgs/hackvm.img
mkdir -p mnt
sudo mount /dev/nbd0p1 mnt
```

## Umount
```
sudo umount mnt
sudo qemu-nbd --disconnect /dev/nbd0
```


# For PCI passthrough

Host cmdline:  intel_iommu=on video=efifb:off,vesafb:off log_buf_len=16M cma=128M@0-4G