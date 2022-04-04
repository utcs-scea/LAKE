CC=g++

all: fs_bench

OBJECTS := main.o Timer.o Benchmark.o

$(OBJECTS): %.o : %.cpp
	$(CC) -c $< -o $@

fs_bench: $(OBJECTS)
	$(CC) $(OBJECTS) -o $@

clean:
	rm -f *.o fs_bench
