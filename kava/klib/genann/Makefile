klib_genann = kgenann

obj-m += $(klib_genann).o
$(klib_genann)-y += genann.o

CORE := ../../core/kern
COMMON := ..
core-y += $(CORE)/init.o \
			$(CORE)/command_handler.o \
			$(CORE)/metadata_map.o \
			$(CORE)/shadow_thread_pool.o \
			$(CORE)/endpoint.o \
			$(CORE)/chan_file_poll.o \
			$(CORE)/chan_nl_socket.o \
			$(COMMON)/control.o

$(klib_genann)-y += $(core-y)

ccflags-y += -I. -I$(src)/../../include -march=native -mhard-float -msse
