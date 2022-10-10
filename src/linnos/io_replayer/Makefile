all: replayer

replayer: replay_failover.c
	gcc @^ -o replayer -lpthread

clean:
	rm -rf *.o  replayer
