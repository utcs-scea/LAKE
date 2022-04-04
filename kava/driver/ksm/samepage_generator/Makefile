CC=gcc
CFLAGS=-g -O0
CFILES=samepage_generator.c
INCLUDE=-I../kava_ksm/cu/xxhash/cpu/
EXEC=generator

all: clean compile

LIB_XXHASH := -L../kava_ksm/cu/xxhash/cpu -l:libxxhash.a

compile:
	$(CC) $(CFLAGS) $(INCLUDE) $(CFILES) -o $(EXEC) $(LIB_XXHASH)

clean:
	rm -rf *.o
	rm -rf $(EXEC)
