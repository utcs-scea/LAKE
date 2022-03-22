all: 
	nvcc -o main main.cpp prediction_gpu.cu prediction_gpu_batch.cu weights.cpp prediction.c -O3

clean: 
	  $(RM) prediction