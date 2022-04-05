#ifndef PACKET_COLLECTOR_H
#define PACKET_COLLECTOR_H

#define SERVER_PORT 9877
#define CLIENT_PORT 9878
#define BUF_SIZE 4096
#define IP_HEADER_SIZE 20
#define UDP_HEADER_SIZE 8
#define DEFAULT_BATCH_SIZE 1024
// in milliseconds
#define DEFAULT_BATCH_WAIT 100
#define DEFAULT_INPUT_THROUGHPUT 0

#define RESULT_ERROR -1
#define RESULT_DROP -2
#define RESULT_FORWARD -3
#define RESULT_UNSET -4

#define HEADER_ONLY
#define RETURN_PACKETS_IMMEDIATELY
#define SET_INCOMING_THROUGHPUT

typedef struct _packet
{
    char ip[IP_HEADER_SIZE];
    char udp[UDP_HEADER_SIZE];
#ifndef HEADER_ONLY
    char empty[BUF_SIZE - IP_HEADER_SIZE - UDP_HEADER_SIZE - sizeof(char*)];
#endif /* HEADER_ONLY */
    int size;
    char *payload;
} packet;

void free_packets(void);
int get_packets(packet* p);
int set_batch_size(int s);
int get_batch_size(void);
int set_batch_wait(int s);
int get_batch_wait(void);
int set_input_throughput(int t);
int get_input_throughput(void);

#endif /* PACKET_COLLECTOR_H */
