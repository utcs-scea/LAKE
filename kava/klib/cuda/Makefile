klib_cuda = kcuda

obj-m += $(klib_cuda).o
$(klib_cuda)-y += cuda.o

CORE := ../../core/kern
COMMON := ..
core-y += $(CORE)/init.o \
			$(CORE)/command_handler.o \
			$(CORE)/contention.o \
			$(CORE)/metadata_map.o \
			$(CORE)/shadow_thread_pool.o \
			$(CORE)/endpoint.o \
			$(CORE)/chan_file_poll.o \
			$(CORE)/chan_nl_socket.o \
			$(COMMON)/control.o

$(klib_cuda)-y += $(core-y)

ccflags-y += -I. -I$(src)/../../include -O3 -g
