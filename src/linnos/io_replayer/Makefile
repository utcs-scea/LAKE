all: replayer

replayer: replayer.c io_ops.c
	gcc $^ -o replayer -lpthread

clean:
	rm -rf *.o  replayer
