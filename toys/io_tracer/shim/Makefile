all:
	gcc -g3 -Wall -fPIC -shared -o shim.so reads.cpp -ldl -lpthread -Wl,--version-script -Wl,VERSION.txt