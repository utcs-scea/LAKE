klib_mvnc = kmvnc

obj-m += $(klib_mvnc).o
$(klib_mvnc)-y += mvnc_nw_guestlib.o

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

$(klib_mvnc)-y += $(core-y)

ccflags-y += -I. -I$(src)/../../include -O3 -g
