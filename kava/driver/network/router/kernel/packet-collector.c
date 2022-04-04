#include <linux/random.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include "packet-collector.h"

int batch_size = DEFAULT_BATCH_SIZE;
int batch_wait = DEFAULT_BATCH_WAIT;
int input_throughput = DEFAULT_INPUT_THROUGHPUT;

#ifdef RETURN_PACKETS_IMMEDIATELY
packet *random_buf = NULL;
#endif

void free_packets(void)
{
#ifdef RETURN_PACKETS_IMMEDIATELY
    if (random_buf != NULL) {
        kfree(random_buf);
    }
#endif
}

#ifdef RETURN_PACKETS_IMMEDIATELY
static void generate_random_packets(void)
{
    int i, j;
    free_packets();
    random_buf = (packet*)kmalloc(sizeof(packet) * get_batch_size(), GFP_KERNEL);

    for (i = 0; i < get_batch_size(); i++) {
        for (j = 0; j < IP_HEADER_SIZE; j++)
            random_buf[i].ip[j] = get_random_int() % 256;
        for (j = 0; j < UDP_HEADER_SIZE; j++)
            random_buf[i].udp[j] = get_random_int() % 256;
        random_buf[i].payload = (char *)kmalloc(BUF_SIZE * sizeof(char), GFP_KERNEL);
        for (j = 0; j < BUF_SIZE; j++)
            random_buf[i].payload[j] = get_random_int() % 256;
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

int set_input_throughput(int t)
{
    if (t > 0)
        input_throughput = t;
    return input_throughput;
}

int get_input_throughput(void)
{
    return input_throughput;
}

int get_packets(packet* p)
{
#ifdef RETURN_PACKETS_IMMEDIATELY
    if (random_buf == NULL) generate_random_packets();

#ifdef SET_INCOMING_THROUGHPUT
    /**
     * When the input throughput is less than the processing throughput, the
     * firewall always has to wait for the last packet in the batch to arrive.
     * Equivalently, we wait for X seconds to emulate the gap between the first
     * packet and the last one. (The last packet waits for 0 second, and the
     * first one waits for X seconds.)
     * X = batch_size * 64 / input_throughput / (1<<30)
     *
     * When the input throughput is larger than the processing throughput, a
     * new batch is ready every time when the last batch processing is done.
     * The rest of the incoming packets will be congested. The average latency
     * can be set to infinite.
     */
    //udelay(get_batch_size() * 64 / input_throughput / (1 << 30) * 1000000);
    if (input_throughput > 0) {
        long t = (long)get_batch_size() * 1000000 / input_throughput / (1 << 24);
        udelay(t);
        //printk(KERN_INFO "Add latency %ld\n", t);
    }
#endif // SET_INCOMING_THROUGHPUT

    memcpy(p, random_buf, sizeof(packet)*get_batch_size());
    return get_batch_size();
#endif // RETURN_PACKETS_IMMEDIATELY
}
