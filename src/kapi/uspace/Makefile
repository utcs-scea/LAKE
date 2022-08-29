ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

LIBS=$(shell pkg-config --libs libnl-3.0) -L/usr/local/cuda/lib64 -lcuda -lnvidia-ml
CFLAGS=$(shell pkg-config --cflags libnl-3.0) -I$(ROOT_DIR)/../include

all: lake_uspace

lake_uspace: *.c *.cpp
	nvcc $(CFLAGS) $^ -o $@ $(LIBS)

clean:
	rm -f lake_uspace