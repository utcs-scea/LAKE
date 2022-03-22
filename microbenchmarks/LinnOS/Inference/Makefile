all: 
	nvcc -o main main.cpp prediction_gpu.cu prediction_gpu_batch.cu weights.cpp prediction.cpp -O3

clean: 
	  $(RM) prediction