### Source code of paper [*Machine Learning for Load Balancing in the Linux Kernel*](https://doi.org/10.1145/3409963.3410492)

Prerequisites:

- [BCC](https://github.com/iovisor/bcc)
- [Tensorflow](https://www.tensorflow.org/)


Dump load balance data:
``` bash
sudo ./dump_lb.py -t tag --old
```
> use `--old` with original kernel without test flag


Automated training and evaluation:
```bash 
cd training
./automate.py -t tag1 tag2 tag3... -o model_name
```

Preprocessing: `training/prep.py`

Training: `training/keras_lb.py`
