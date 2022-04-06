klib_darknet = kdarknet

obj-m += $(klib_darknet).o
$(klib_darknet)-y += darknet.o

CORE := ../../core/kern
COMMON := ..
core-y += $(CORE)/init.o \
			$(CORE)/command_handler.o \
			$(CORE)/metadata_map.o \
			$(CORE)/shadow_thread_pool.o \
			$(CORE)/endpoint.o \
			$(CORE)/chan_file_poll.o \
			$(COMMON)/control.o

$(klib_darknet)-y += $(core-y)

ccflags-y += -I. -I$(src)/../../include
