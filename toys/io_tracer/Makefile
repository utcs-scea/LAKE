touchblk: dummy.dat
	gcc touchblk.c -o touchblk

clean:
	rm -f touchblk
	rm -f shim.so

run:
	./touchblk

run_shim: touchblk shim.so dummy.dat 
	LD_PRELOAD="./shim.so" ./touchblk

bpf:
	sudo python3 fblktrace.py

dummy.dat:
	dd if=/dev/urandom of=dummy.dat bs=1024 count=1024

drop_cache:
	echo 1 | sudo tee /proc/sys/vm/drop_caches

disable_readahead:
	@echo "Replace vda with your block device and run this command"
	@echo "echo 0 | sudo tee /sys/block/vda/queue/read_ahead_kb"

shim.so:
	g++ -Wall -fpic -shared shim/*.cpp -ldl -lpthread -o shim.so -Wl,--version-script=shim/VERSION.txt