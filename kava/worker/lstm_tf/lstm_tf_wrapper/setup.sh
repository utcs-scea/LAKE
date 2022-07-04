gcc $(python3.6-config --cflags) -fPIE -c c_wrapper.c -o c_wrapper.o
gcc $(python3.6-config --cflags) -fPIE -c main.c -o main.o
gcc c_wrapper.o main.o $(python3.6-config --libs) -o test
