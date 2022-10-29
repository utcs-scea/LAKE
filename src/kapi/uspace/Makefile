ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

LIBS=$(shell pkg-config --libs libnl-3.0) -L/usr/local/cuda/lib64 -lcuda -lnvidia-ml
CFLAGS=$(shell pkg-config --cflags libnl-3.0) -I$(ROOT_DIR)/../include

python_version_full := $(wordlist 2,4,$(subst ., ,$(shell python3 --version 2>&1)))
python_version_major := $(word 1,${python_version_full})
python_version_minor := $(word 2,${python_version_full})

#This string is python3.10 or whatever version you have.
#if this doesnt work, hard code it by commenting it out
#and adding something like:
#pyver := "python3.8"   (replace with your version)
pyver := python$(python_version_major).$(python_version_minor)

#after python3.8 we need the --embed flag -.-
$(if $(shell [ $(python_version_minor) -ge 8 ] && echo "OK"), \
    $(eval embf := --embed), \
	$(eval embf := ))

PY_LINKER_FLAGS = $(shell $(pyver)-config --libs $(embf))
PY_COMPILE_FLAGS = $(shell $(pyver)-config --cflags $(embf))

all: lake_uspace

kleio/py_wrapper.o:
	cd kleio && make

lake_uspace: *.c *.cpp kleio/py_wrapper.o
	nvcc $(CFLAGS) $(PY_COMPILE_FLAGS) $^ -o $@ $(LIBS)  $(PY_LINKER_FLAGS)

clean:
	rm -f lake_uspace