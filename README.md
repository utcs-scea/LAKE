# HAK


task kavad:15916 blocked for more than 120 seconds.
[10996.862116]       Tainted: P           OE     4.19.237 #2
[10996.862189] "echo 0 > /proc/sys/kernel/hung_task_timeout_secs" disables this message.



# To play with mllb training, which uses tensorflow and python


## To create (done only once):

```
python3.7 -m venv .hack
. .hack/bin/activate
pip install -r requirements.txt
```

## To activate
`. .hack/bin/activate`


## Dumping training data is a PITA

The command is :
`sudo -E env PATH=${PATH} python3 dump_lb.py -t tag2`
but there are dependencies that are not easy to install. Zemaitis has them in there.
This includes a bcc python package that was manually fixed and needs to be copied into
the python path. It has been copied to toys/. The command used to install it is

`cp -r bcc-python3-module/bcc .hack/lib/python3.7/site-packages/`

## Training

First there is some prep to do (replace 2 with the number in the input file)
`python prep.py -t tag2`
then
`python keras_lb.py -o model2 -t`.
Add `-g` to use a GPU.





## eBPF

sudo apt-get install libreadline-dev binutils-dev clang-11
Go to linux-5.15.65/tools/bpf, make and make install 

Clang might not be in path, do something like 
sudo ln -s /usr/bin/clang-11 /usr/bin/clang

wget https://github.com/libbpf/libbpf/archive/refs/tags/v1.0.0.tar.gz
untar, cd src, make

sudo mount -t tracefs nodev /sys/kernel/tracing

sudo apt-get install libelf-dev libdwarf-dev # some dependency install
git clone https://github.com/acmel/dwarves.git
cd dwarves/
git checkout tags/v1.19 -b V1.19-branch # take the latest version
mkdir build
cd build
cmake -D__LIB=lib ..
sudo make install
sudo /sbin/ldconfig -v

compile kernel, boot into it

