# gcc $(python3-config --cflags) -fPIE -c c_wrapper.c -o c_wrapper.o
# gcc c_wrapper.o $(python3-config --libs) -o test

CC=gcc
CXX=g++
LINKER=gcc

python_version_full := $(wordlist 2,4,$(subst ., ,$(shell python3 --version 2>&1)))
python_version_major := $(word 1,${python_version_full})
python_version_minor := $(word 2,${python_version_full})

#This string is python3.10 or whatever version you have.
#if this doesnt work, hard code it by commenting it out
#and adding something like:
#pyver := "python3.8"   (replace with your version)
pyver := "python$(python_version_major).$(python_version_minor)"

#after python3.8 we need the --embed flag -.-
$(if $(shell [ $(python_version_minor) -ge 8 ] && echo "OK"), \
    $(eval embf := --embed), \
	$(eval embf := ))

PY_LINKER_FLAGS = $(shell $(pyver)-config --libs $(embf))
#PY_COMPILE_FLAGS = $(shell $(pyver)-config --cflags $(embf))

PY_COMPILE_FLAGS = -I/usr/include/$(pyver) -DNDEBUG -O2 -lcrypt -ldl  -lm

current_dir = $(shell pwd)
model:=/../../../kleio/lstm_page_539
fullp:=$(realpath  $(current_dir)$(model))
CFLAGS += -D__MODELPATH__=\"$(fullp)\"
CFLAGS += -D__INCPATH__=\"$(current_dir)\"
pypackages:=$(realpath  $(current_dir)/../../../../lakevenv/lib/python3.10/site-packages/)
CFLAGS += -D__INCPATH2__=\"$(pypackages)\"


all:
	gcc -c py_wrapper.c -o py_wrapper.o $(CFLAGS) $(PY_COMPILE_FLAGS) -fPIE  $(PY_LINKER_FLAGS)
	gcc -c wrapper_test.c -o wrapper_test.o $(CFLAGS) $(PY_COMPILE_FLAGS) -fPIE $(PY_LINKER_FLAGS)
	gcc py_wrapper.o wrapper_test.o  -o test $(PY_LINKER_FLAGS)
	#nvcc $(PY_COMPILE_FLAGS) wrapper_test.c py_wrapper.c -o test2 $(CFLAGS)  $(PY_LINKER_FLAGS)

clean:
	rm -rf *.o