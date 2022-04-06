#ifndef __MNIST_H__
#define __MNIST_H__

#ifndef __KERNEL__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#endif /* __KERNEL__ */

// Consider only 0s and 1s
#define LABELS_SIZE 2

typedef struct mnist_data {
	unsigned char data[28][28];
	unsigned int label;
} mnist_data;

typedef struct mnist_data_ann {
	double data[28*28];
	double label;
} mnist_data_ann;

int loadMNISTData(const char *image_filename, const char *label_filename,
	mnist_data **data, unsigned int *count);

// Load with labels less than LABELS_SIZE
int loadMNISTDataUpTo(const char *image_filename, const char *label_filename,
    mnist_data **data, unsigned int *count);

#endif /* __MNIST_H__ */
