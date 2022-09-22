all: replayer

replayer: replayer.o
	gcc -o replayer replay.o -lpthread -g

writer: writer.c
	gcc -o writer writer.c -pthread -g

replayer.o:
	gcc -c replay.c -g

replayer_fail: replayer_fail.o
	gcc -o replayer_fail replay_failover.o -lpthread -g

replayer_fail.o:
	gcc -c replay_failover.c -g

replayer_back: replayer_back.o
	gcc -o replayer_back replay_back.o -lpthread -g

replayer_back.o:
	gcc -c replay_back.c -g

replayer_fore: replayer_fore.o
	gcc -o replayer_fore replay_fore.o -lpthread -g

replayer_fore.o:
	gcc -c replay_fore.c -g

replayer_clone: replayer_clone.o
	gcc -o replayer_clone replay_clone.o -lpthread -g

replayer_clone.o:
	gcc -c replay_clone.c -g

clean:
	rm -rf *.o replayer writer replayer_fail
