CC = gcc
CFLAGS = "-I../include"

all: kavactrl

kavactrl: kavactrl.c
	$(CC) -o $@ $@.c $(CFLAGS) `pkg-config --cflags glib-2.0`

clean:
	rm kavactrl
