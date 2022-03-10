# HAK




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