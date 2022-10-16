all:
	gcc -g3 -Wall -fPIC -shared -o shim.so reads.c lwrb.c tracer.c -ldl -lpthread -Wl,--version-script -Wl,VERSION.txt