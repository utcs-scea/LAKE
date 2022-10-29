# gcc $(python3-config --cflags) -fPIE -c c_wrapper.c -o c_wrapper.o
# gcc c_wrapper.o $(python3-config --libs) -o test

CC=gcc
CXX=g++
LINKER=gcc

python_version_full := $(wordlist 2,4,$(subst ., ,$(shell python --version 2>&1)))
python_version_major := $(word 1,${python_version_full})
python_version_minor := $(word 2,${python_version_full})

#This string is python3.10 or whatever version you have.
#if this doesnt work, hard code it by commenting it out
#and adding something like:
#pyver := "python3.8"   (replace with your version)
pyver := "python$(python_version_major).$(python_version_minor)"

PY_LINKER_FLAGS = $(shell $(pyver)-config --embed --libs)
PY_COMPILE_FLAGS = $(shell $(pyver)-config --cflags --embed)

current_dir = $(shell pwd)
model:="/../../../kleio/lstm_page_539"
fullp:=$(current_dir)$(model)
CFLAGS += -D__MODELPATH__=\"$(fullp)\"

all:
	gcc -c py_wrapper.c -o py_wrapper.o $(CFLAGS) $(PY_COMPILE_FLAGS) -fPIE  $(PY_LINKER_FLAGS)
	gcc -c wrapper_test.c -o wrapper_test.o $(CFLAGS) $(PY_COMPILE_FLAGS) -fPIE $(PY_LINKER_FLAGS)
	gcc py_wrapper.o wrapper_test.o  -o test $(PY_LINKER_FLAGS)

clean:
	rm -rf *.o