#ifndef PACKET_COLLECTOR_H
#define PACKET_COLLECTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pthread.h>

#define SERVER_PORT 9877
#define CLIENT_PORT 9878
#define BUF_SIZE 4096
#define IP_HEADER_SIZE 20
#define UDP_HEADER_SIZE 8
#define DEFAULT_BATCH_SIZE 1024
// in milliseconds
#define DEFAULT_BATCH_WAIT 100

#define RESULT_ERROR -1
#define RESULT_DROP -2
#define RESULT_FORWARD -3
#define RESULT_UNSET -4

#define HEADER_ONLY
#define RETURN_PACKETS_IMMEDIATELY

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

typedef struct _udpc
{
    int fd; // sock descriptor
    struct sockaddr_in sa; // server address
} udpc;

int get_packets(int sockfd, packet* p);
int send_packets(udpc client, packet* p, int num_packets, int* a);
int init_server_socket();
udpc init_client_socket();
int set_batch_size(int s);
int get_batch_size();
int set_batch_wait(int s);
int get_batch_wait();

#endif /* PACKET_COLLECTOR_H */
