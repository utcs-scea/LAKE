CC=g++

all: fs_bench

OBJECTS := main.o Timer.o Benchmark.o

$(OBJECTS): %.o : %.cpp
	$(CC) -c $< -o $@

fs_bench: $(OBJECTS)
	$(CC) $(OBJECTS) -o $@

#drop_cache:
#	$(CC) drop_cache.cpp -o drop_cache
#	@echo "Setting owner to root and setting uid, need root"
#	sudo chown root drop_cache
#	sudo chmod u+s drop_cache

clean:
	rm -f *.o fs_bench drop_cache
