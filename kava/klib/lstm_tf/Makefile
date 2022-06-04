klib_lstm_tf = lstm_tf

obj-m += $(klib_lstm_tf).o
$(klib_lstm_tf)-y += lstm_tf_nw_guestlib.o

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

$(klib_lstm_tf)-y += $(core-y)

ccflags-y += -I. -I$(src)/../../include -O3 -g
