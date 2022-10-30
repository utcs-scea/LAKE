all:
	gcc util_killable.c -o kutil_kill
	gcc util.c -o util
	g++ cpu_gpu.cpp -o cpu_gpu -lnvidia-ml -lpthread -I/usr/local/cuda/include
	g++ -o ReadWriteData ReadWriteData.cpp
	
