#include "packet-collector.h"

int batch_size = DEFAULT_BATCH_SIZE;
int batch_wait = DEFAULT_BATCH_WAIT;

#ifdef RETURN_PACKETS_IMMEDIATELY
packet *random_buf = NULL;
#endif

int init_server_socket()
{
    int sockfd;
    struct sockaddr_in servaddr;

    if ((sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0) {
        perror("socket");
        return -1;
    }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family      = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port        = htons(SERVER_PORT);

    if (bind(sockfd, (struct sockaddr *) &servaddr, sizeof(servaddr)) < 0) {
        perror("bind");
        return -1;
    }

    return sockfd;
}

udpc init_client_socket()
{
    udpc client;

    if ((client.fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0) {
        perror("socket");
        return client;
    }

    memset(&client.sa, 0, sizeof(client.sa));
    client.sa.sin_family = AF_INET;
    client.sa.sin_port = htons(CLIENT_PORT);
    if (inet_aton("127.0.0.1", &client.sa.sin_addr) == 0) {
        perror("inet_aton");
        client.fd = -1;
        return client;
    }

    return client;
}

#ifdef RETURN_PACKETS_IMMEDIATELY
void generate_random_packets()
{

    if (random_buf != NULL) {
        free(random_buf);
    }
    random_buf = (packet*)malloc( sizeof(packet) * get_batch_size());

    for (int i = 0; i < get_batch_size(); i++) {
        for (int j = 0; j < IP_HEADER_SIZE; j++) random_buf[i].ip[j] = (char)rand();
        for (int j = 0; j < UDP_HEADER_SIZE; j++) random_buf[i].udp[j] = (char)rand();
        random_buf[i].payload = (char *)malloc(BUF_SIZE * sizeof(char));
        for (int j = 0; j < BUF_SIZE; j++) random_buf[i].payload[j] = (char)rand();
    }
}
#endif /* RETURN_PACKETS_IMMEDIATELY */

int set_batch_size(int s)
{
    if (s > 0) {
        batch_size = s;
#ifdef RETURN_PACKETS_IMMEDIATELY
        generate_random_packets();
#endif /* RETURN_PACKETS_IMMEDIATELY */
    }
    return batch_size;
}

int get_batch_size()
{
    return batch_size;
}

int set_batch_wait(int s)
{
    if (s > 0) {
        batch_wait = s;
    }
    return batch_wait;
}

int get_batch_wait()
{
    return batch_wait;
}

void *timer(void *data)
{
    usleep(batch_wait * 1000);
    *(int *)data = 1;
    return 0;
}

int get_packets(int sockfd, packet* p)
{
#ifdef RETURN_PACKETS_IMMEDIATELY
    if (random_buf == NULL) generate_random_packets();

    memcpy(p, random_buf, sizeof(packet)*get_batch_size());
    return get_batch_size();
#endif

    int timeout = 0;
    pthread_t thread;

    struct timeval tout;
    tout.tv_sec = 0;
    tout.tv_usec = 0;

    fd_set rfds;
    FD_ZERO(&rfds);

    pthread_create(&thread, NULL, timer, &timeout);

    int i;
    for (i = 0; i < batch_size;) {
        FD_SET(sockfd, &rfds);
        if (select(sockfd + 1, &rfds, NULL, NULL, &tout) > 0) {
            p[i].size = recv(sockfd, p[i].payload, BUF_SIZE, 0);
            if (p[i].size > IP_HEADER_SIZE + UDP_HEADER_SIZE) {
                memcpy(p[i].ip, p[i].payload, IP_HEADER_SIZE);
                memcpy(p[i].udp, &p[i].payload[IP_HEADER_SIZE], UDP_HEADER_SIZE);
                i++;
            }
        }
        if (timeout == 1) break;
    }
    pthread_cancel(thread);
    return i;
}

int send_packets(udpc client, packet* p, int num_packets, int* a)
{
#ifndef RETURN_PACKETS_IMMEDIATELY
    for (int i = 0; i < num_packets && a[i] >= 0; i++) {
        sendto(client.fd, p[i].payload, p[i].size, 0, (struct sockaddr*)&client.sa, sizeof(client.sa));
    }
#endif /* RETURN_PACKETS_IMMEDIATELY */
    return 0;
}
