# gcc $(python3-config --cflags) -fPIE -c c_wrapper.c -o c_wrapper.o
# gcc c_wrapper.o $(python3-config --libs) -o test

CC=gcc
CXX=g++
LINKER=gcc

PY_COMPILE_FLAGS = $(shell python3.6-config --cflags)
PY_LINKER_FLAGS = $(shell python3.6-config --libs)

all:
	gcc $(PY_COMPILE_FLAGS) -fPIE -c py_wrapper.c -o py_wrapper.o
	gcc $(PY_COMPILE_FLAGS) -fPIE -c wrapper_test.c -o wrapper_test.o
	gcc py_wrapper.o wrapper_test.o $(PY_LINKER_FLAGS) -o test

clean:
	rm -rf *.o
