#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void *fibo(void *x);

int main(int argc, char *argv[]) {
    pthread_attr_t attr;
    unsigned long count;
    void *ret;

    if (argc != 2) {
		fprintf(stderr,"usage: pthreads <integer value>\n");
		return -1;
	}

	count = atoi(argv[1]);

	if (count < 1) {
		fprintf(stderr,"%lu must be>= 1\n", count);
		return -1;
	}

    ret = fibo((void *)count);

    printf("fibo out: %lu\n", (unsigned long)ret);
    return 0;
}


void *fibo(void *x) {
    unsigned long n = (unsigned long)x;
    if (n <= 1) {
        return (void *)1;
    }
    void *add_1, *add_2;

    pthread_attr_t attr;
    pthread_t thread1, thread2;

    /* printf("fibo %lu\n", n); */
    pthread_attr_init(&attr);
    pthread_create(&thread1, &attr, fibo, (void *)(n-1));
    pthread_create(&thread2, &attr, fibo, (void *)(n-2));

    pthread_join(thread1, &add_1);
    pthread_join(thread2, &add_2);

    return (void *)((unsigned long)add_1 + (unsigned long)add_2);
}
