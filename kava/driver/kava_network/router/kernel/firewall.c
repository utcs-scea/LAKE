#include <linux/slab.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include "router.h"

#ifdef FIREWALL

typedef struct _rule {
  uint32_t src_ip;
  uint32_t dst_ip;
  uint16_t src_port;
  uint16_t dst_port;
  uint8_t proto;
  int8_t action;
} __attribute__((aligned(16)))
rule;

int num_rules = 100;

int set_num_rules(int s) {
	if (s > 0) {
		num_rules = s;
	}
	return num_rules;
}

int get_num_rules() {
	return num_rules;
}

/* Global vars for firewall */
//unsigned long *h_rule_hashes;
//unsigned long *d_rule_hashes;
rule *h_rules;
CUdeviceptr d_rules;
int h_num_rules;

/**
 * Generates a set of firewall rules
 */
void generate_rules(int num_rules, rule* rules) 
{
	// Generate some random rules
	int i;
	unsigned int r;
	for (i = 0; i < num_rules; i++) {

		/**
		 *	Pick a protocol:
		 *  * 6%,  TCP 75%,   UDP 14%,   ICMP 4%,   Other 1%
		 */
		r = get_random_int() % 101;
		if (r <= 75) {
			rules[i].proto = 6; // TCP
		} else if (r <= 89) {
			rules[i].proto = 17; // UDP
		} else if (r <= 93) {
			rules[i].proto = 1; // ICMP
		} else if (r <= 99) {
			rules[i].proto = 0; // *
		} else {
			rules[i].proto = get_random_int() % 255; // Random other
		}

		/**
		 * Pick a source port:
		 *	* 98%,   range 1%,   single value 1%
		 * (note: ignoring ranges for now)
		 */
		r = get_random_int() % 101;
		if (r <= 98) {
			rules[i].src_port = 0; // *
		} else {
			rules[i].src_port = get_random_int() % 65535; // Random port
		}

		/**
		 * Pick a dest port:
		 *	* 0%,   range 4%,   80 6.89%,   21 5.65%,   23 4.87%,   443 3.90%,   8080 2.25%,   139 2.16%,   Other
		 * (note: ignoring ranges for now)
		 */
		r = get_random_int() % 10001;
		if (r <= 689) {
			rules[i].dst_port = 80;
		} else if (r <= 1254) {
			rules[i].dst_port = 21;
		} else if (r <= 1741) {
			rules[i].dst_port = 23;
		} else if (r <= 2131) {
			rules[i].dst_port = 443;
		} else if (r <= 2356) {
			rules[i].dst_port = 8080;
		} else if (r <= 2572) {
			rules[i].dst_port = 139;
		} else {
			rules[i].dst_port = get_random_int() % 65535; // Random port
		}

		/**
		 * Pick a src IP:
		 *	* 95%,   range 5%
		 * (note: ignoring ranges for now)
		 */
		rules[i].src_ip = get_random_int() % 4294967295;

		/**
		 * Pick a dst IP:
		 *	single ip 45%,   range 15%,   Class B 10%,   Class C 30%
		 * (note: ignoring ranges for now)
		 */
		r = get_random_int() % 100;
		if (r <= 45) {
			rules[i].dst_ip = get_random_int() % 4294967295;
		} else {
			rules[i].dst_ip = 0; // using wildcard instead of range
		}

		rules[i].action = RESULT_DROP;
	}

	/*inet_pton(AF_INET, "123.123.123.123", &(rules[0].src_ip));
	inet_pton(AF_INET, "210.210.210.210", &(rules[0].dst_ip));
	rules[0].src_port = 1234;
	rules[0].dst_port = 4321;
	rules[0].proto = 17;
	rules[0].action = RESULT_DROP;*/

	/*rules[0].src_ip = 0;
	rules[0].dst_ip = 0;
	rules[0].src_port = 0;
	rules[0].dst_port = 0;
	rules[0].proto = 17;
	rules[0].action = RESULT_DROP;*/

}

/**
 * Firewall-specific setup. This will be called a single time by router.cu 
 * before the kernel function runs for the first time
 */
void setup_gpu()
{
	int rules_size;

	h_num_rules = num_rules;
    rules_size = h_num_rules * sizeof(rule);
	h_rules = (rule*)kava_alloc(rules_size);
	check_error(cuMemAlloc((CUdeviceptr*) &d_rules, rules_size), "cuMemAlloc d_rules", __LINE__);

	generate_rules(h_num_rules, h_rules);

	// Copy firewall rules to GPU so the kernel function can use them
	check_error(cuMemcpyHtoD(d_rules, h_rules, rules_size), "cuMemcpyHtoD (d_rules h_rules)", __LINE__);
}


/**
 * Firewall-specific teardown. This will be called a single time by router.cu 
 * after the kernel function runs last time
 */
void teardown()
{
	kava_free(h_rules);
	cuMemFree(d_rules);
}

/**
 * A CPU-only sequential version of the firewall packet processing function
 */
void process_packets_sequential(packet *p, int *results, int num_packets)
{
	int packet_index;
	int i;
	for (packet_index = 0; packet_index < get_batch_size(); packet_index++) {
		struct iphdr *ip_hdr = (struct iphdr*)p[packet_index].ip;

		uint16_t sport, dport;
		if (ip_hdr->protocol == 17) {
			struct udphdr *udp_hdr = (struct udphdr*)p[packet_index].udp;
			sport = ntohs(udp_hdr->source);
			dport = ntohs(udp_hdr->dest);
		} else if (ip_hdr->protocol == 6) {
			struct tcphdr *tcp_hdr = (struct tcphdr*)p[packet_index].udp; // FIXME
			sport = ntohs(tcp_hdr->source);
			dport = ntohs(tcp_hdr->dest);
		} else {
			sport = 0;
			dport = 0;
		}

		// TODO: make this handle other protocols (TCP)
		for (i = 0; i < h_num_rules; i++) {
			if ((h_rules[i].src_ip == 0 || h_rules[i].src_ip == ip_hdr->saddr) &&
				(h_rules[i].dst_ip == 0 || h_rules[i].dst_ip == ip_hdr->saddr) &&
				(h_rules[i].src_port == 0 || h_rules[i].src_port == sport) &&
				(h_rules[i].dst_port == 0 || h_rules[i].dst_port == dport) &&
				(h_rules[i].proto == 0 || h_rules[i].proto == ip_hdr->protocol))
			{
				results[packet_index] = h_rules[i].action;
				break;
			}
			else
			{
				results[packet_index] = RESULT_FORWARD;
			}
		}
	}
}

/**
 * Firewall-specific setup. This will be called a single time by router.cu
 * before the sequential CPU function runs for the first time
 */
void setup_sequential()
{
    int rules_size;

	h_num_rules = num_rules;
	rules_size = h_num_rules * sizeof(rule);
	h_rules = (rule*)kmalloc(rules_size, GFP_KERNEL);

	generate_rules(h_num_rules, h_rules);
}

#endif /* FIREWALL */
