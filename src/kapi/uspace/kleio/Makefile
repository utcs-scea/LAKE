# gcc $(python3-config --cflags) -fPIE -c c_wrapper.c -o c_wrapper.o
# gcc c_wrapper.o $(python3-config --libs) -o test

CC=gcc
CXX=g++
LINKER=gcc

PY_COMPILE_FLAGS = $(shell python3.10-config --cflags --embed)
PY_LINKER_FLAGS = $(shell python3.10-config --embed --libs)

all:
	gcc -c py_wrapper.c -o py_wrapper.o $(PY_COMPILE_FLAGS) -fPIE  $(PY_LINKER_FLAGS)
	gcc -c wrapper_test.c -o wrapper_test.o $(PY_COMPILE_FLAGS) -fPIE $(PY_LINKER_FLAGS)
	gcc py_wrapper.o wrapper_test.o  -o test $(PY_LINKER_FLAGS)

clean:
	rm -rf *.o
