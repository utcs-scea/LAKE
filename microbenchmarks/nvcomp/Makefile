all:
	nvcc main.cpp -lnvcomp -llz4 -o main -O3

run:
	CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib ./main