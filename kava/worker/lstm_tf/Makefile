ifdef RELEASE
KAVA_RELEASE=yes
else ifdef R
KAVA_RELEASE=yes
endif

ifdef KAVA_RELEASE
KAVA_RELEASE_FLAG=-DAVA_RELEASE -DNDEBUG -O2 -flto
else
KAVA_RELEASE_FLAG=-O0 -g
endif

ifdef GPU
KAVA_HAS_GPU=yes
else ifdef G
KAVA_HAS_GPU=yes
endif

ifdef KAVA_HAS_GPU
KAVA_HAS_GPU_FLAG=-DKAVA_HAS_GPU
else
KAVA_HAS_GPU_FLAG=
endif

LSTM_TF_ROOT?=$(abspath ./)

CC=gcc
CXX=g++
LINKER=gcc

override CFLAGS+=$(KAVA_RELEASE_FLAG) -fmax-errors=25 -I$(LSTM_TF_ROOT) -I$(LSTM_TF_ROOT)/lstm_tf_wrapper -I.. `pkg-config --cflags glib-2.0 libnl-3.0` -D_GNU_SOURCE -Wall -Wno-unused-result -Werror=implicit -D_FILE_OFFSET_BITS=64

override LIBS+=-pthread -lrt -ldl

override CXXFLAGS+= -fpermissive

WORKER_LIBS+=`pkg-config --libs glib-2.0 libnl-3.0` -L/usr/local/lib

all: worker

vpath %.c ../
vpath %.c ../../core/user/
vpath %.c $(LSTM_TF_ROOT)/lstm_tf_wrapper/

GENERAL_SOURCES_C=command_handler.c metadata_map.c \
				  shadow_thread_pool.c endpoint.c util.c \
				  chan_file_poll.c chan_nl_socket.c
PYTHON_SOURCES_C=c_wrapper.c
WORKER_SPECIFIC_SOURCES=lstm_tf_nw_worker.c
WORKER_SPECIFIC_SOURCES_C=worker.c ../feature.c

GENERAL_OBJECTS_C=$(addprefix objs/,$(GENERAL_SOURCES_C:.c=.o))
PYTHON_WRAPPER_OBJECT_C=$(addprefix objs/,$(PYTHON_SOURCES_C:.c=.o))
WORKER_SPECIFIC_OBJECTS=$(addprefix objs/,$(patsubst %.cpp,%.o,$(WORKER_SPECIFIC_SOURCES:.c=.o)))
WORKER_SPECIFIC_OBJECTS_C=$(addprefix objs/,$(WORKER_SPECIFIC_SOURCES_C:.c=.o))

PY_COMPILE_FLAGS = $(shell python3.7-config --cflags)
PY_LINKER_FLAGS = $(shell python3.7-config --libs)	# This is for python 3.6 and below, for above, use embed

dump:
	echo $(GENERAL_SOURCES_C)
	echo $(GENERAL_OBJECTS_C)
	echo $(WORKER_SPECIFIC_SOURCES)
	echo $(WORKER_SPECIFIC_OBJECTS)
	echo $(WORKER_SPECIFIC_SOURCES_C)
	echo $(WORKER_SPECIFIC_OBJECTS_C)
	echo $(PYTHON_SOURCES_C)
	echo $(PYTHON_WRAPPER_OBJECT_C)

objs/.directory:
	mkdir -p objs
	touch $@

objs/%.o: %.c objs/.directory
	$(CC) -c $(PY_COMPILE_FLAGS) -fPIC -I../../include $(CFLAGS) $(CPPFLAGS) $< -o $@
objs/%.o: %.cpp objs/.directory
	$(CXX) -c $(PY_COMPILE_FLAGS) -fPIC -I../../include $(CXXFLAGS) $(CFLAGS) $(CPPFLAGS) $< -o $@

worker: $(GENERAL_OBJECTS_C) $(WORKER_SPECIFIC_OBJECTS) $(WORKER_SPECIFIC_OBJECTS_C) $(PYTHON_WRAPPER_OBJECT_C)
	$(LINKER) -I../../include $^ $(CFLAGS) $(WORKER_LIBS) $(LIBS) $(PY_LINKER_FLAGS) -o $@

clean:
	-rm -rf worker
	-rm -rf objs

.PHONY: all clean
