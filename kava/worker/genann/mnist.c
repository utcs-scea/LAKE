#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../klib/genann/mnist.h"

static unsigned int mnistBinaryToInt(char *v) {
	int i;
	unsigned int result = 0;

	for (i = 0; i < 4; ++i) {
		result <<= 8;
		result |= (unsigned char)v[i];
	}

	return result;
}

int loadMNISTData(const char *image_filename, const char *label_filename,
	mnist_data **data, unsigned int *count) {

	/*
	* CONSTANTS DEFINED FROM MNIST WEBSITE
	* http://yann.lecun.com/exdb/mnist/
	*/
	const int LABEL_MAGIC = 2049;
	const int IMAGE_MAGIC = 2051;

	const int ROWS = 28;
	const int COLUMNS = 28;

	//const int IMAGE_SIZE = ROWS * COLUMNS;

	int return_code = 0;
	int i;
	char temp[4];

	unsigned int image_count, label_count;
	unsigned int image_dimensions[2];


	// Open image and label files and check for validity.
	FILE *imageFile = fopen(image_filename, "rb");
	FILE *labelFile = fopen(label_filename, "rb");

	if (!imageFile || !labelFile) {
		printf("Error opening image/label files\n");
		return_code = -1;
		goto cleanup;
	}


	// Read magic numbers from the files and make sure they match the expected
	// constant values.
	ssize_t read_ret;
	read_ret = fread(temp, 1, 4, imageFile);
	if (mnistBinaryToInt(temp) != IMAGE_MAGIC) {
		printf("Bad magic number in image file!\n");
		return_code = -2;
		goto cleanup;
	}

	read_ret = fread(temp, 1, 4, labelFile);
	if (mnistBinaryToInt(temp) != LABEL_MAGIC) {
		printf("Bad magic number in label file!\n");
		return_code = -3;
		goto cleanup;
	}


	// Read image/label counts from the files and make sure there's the same
	// number in each file.
	read_ret = fread(temp, 1, 4, imageFile);
	image_count = mnistBinaryToInt(temp);

	read_ret = fread(temp, 1, 4, labelFile);
	label_count = mnistBinaryToInt(temp);

	if (image_count != label_count) {
		printf("The number of labels and images do not match!\n");
		return_code = -4;
		goto cleanup;
	}


	// Get image dimensions from the file and make sure they match the expected
	// constant values.
	for (i = 0; i < 2; ++i) {
		read_ret = fread(temp, 1, 4, imageFile);
		image_dimensions[i] = mnistBinaryToInt(temp);
	}

	if (image_dimensions[0] != ROWS || image_dimensions[1] != COLUMNS) {
		printf("Image dimensions don't match the expected %dx%d", ROWS, COLUMNS);
		return_code = -2;
		goto cleanup;
	}


	// Read image/label data from the files and put in data array
    *count = image_count;
	*data = (mnist_data *)malloc(sizeof(mnist_data) * image_count);

	for (i = 0; i < image_count; ++i) {
		unsigned char read_data[ROWS * COLUMNS];
		mnist_data *d = &(*data)[i];

		read_ret = fread(read_data, 1, ROWS*COLUMNS, imageFile);

        memcpy(d->data, read_data, ROWS*COLUMNS);

        read_ret = fread(temp, 1, 1, labelFile);
        d->label = temp[0];
	}

cleanup:
	if (imageFile) fclose(imageFile);
	if (labelFile) fclose(labelFile);

	return return_code;
}


int loadMNISTDataUpTo(const char *image_filename, const char *label_filename,
    mnist_data **data, unsigned int *count) {

    /*
    * CONSTANTS DEFINED FROM MNIST WEBSITE
    * http://yann.lecun.com/exdb/mnist/
    */
    const int LABEL_MAGIC = 2049;
    const int IMAGE_MAGIC = 2051;

    const int ROWS = 28;
    const int COLUMNS = 28;

    //const int IMAGE_SIZE = ROWS * COLUMNS;

    int return_code = 0;
    int i;
    char temp[4];

    unsigned int image_count, label_count;
    unsigned int image_dimensions[2];


    // Open image and label files and check for validity.
    FILE *imageFile = fopen(image_filename, "rb");
    FILE *labelFile = fopen(label_filename, "rb");

    if (!imageFile || !labelFile) {
        printf("Error opening image/label files\n");
        return_code = -1;
    }


    // Read magic numbers from the files and make sure they match the expected
    // constant values.
 	ssize_t read_ret;
    read_ret = fread(temp, 1, 4, imageFile);
    if (mnistBinaryToInt(temp) != IMAGE_MAGIC) {
        printf("Bad magic number in image file!\n");
        return_code = -2;
    }

    read_ret = fread(temp, 1, 4, labelFile);
    if (mnistBinaryToInt(temp) != LABEL_MAGIC) {
        printf("Bad magic number in label file!\n");
        return_code = -3;
    }


    // Read image/label counts from the files and make sure there's the same
    // number in each file.
    read_ret = fread(temp, 1, 4, imageFile);
    image_count = mnistBinaryToInt(temp);

    read_ret = fread(temp, 1, 4, labelFile);
    label_count = mnistBinaryToInt(temp);

    if (image_count != label_count) {
        printf("The number of labels and images do not match!\n");
        return_code = -4;
    }


    // Get image dimensions from the file and make sure they match the expected
    // constant values.
    for (i = 0; i < 2; ++i) {
        read_ret = fread(temp, 1, 4, imageFile);
        image_dimensions[i] = mnistBinaryToInt(temp);
    }

    if (image_dimensions[0] != ROWS || image_dimensions[1] != COLUMNS) {
        printf("Image dimensions don't match the expected %dx%d", ROWS, COLUMNS);
        return_code = -2;
    }

    unsigned int* labelsArray = (unsigned int*)malloc(sizeof(unsigned int) * image_count);

    // count how many labelas are less than LABELS_SIZE
    unsigned int countUpTo = 0;
    for (i = 0; i < image_count; ++i) {
        read_ret = fread(temp, 1, 1, labelFile);
        labelsArray[i] = temp[0];
        if (temp[0] < LABELS_SIZE)
            ++countUpTo;
    }

    // Read image/label data from the files and put in data array
    *count = countUpTo;
    *data = (mnist_data *)malloc(sizeof(mnist_data) * countUpTo);
    int j = 0;
    for (i = 0; i < image_count; ++i) {
        unsigned char read_data[ROWS * COLUMNS];
        read_ret = fread(read_data, 1, ROWS*COLUMNS, imageFile);

        if (labelsArray[i] >= LABELS_SIZE)
            continue;

        mnist_data *d = &(*data)[j++];
        memcpy(d->data, read_data, ROWS*COLUMNS);

        d->label = labelsArray[i];
    }

    if (imageFile) fclose(imageFile);
    if (labelFile) fclose(labelFile);

    return return_code;
}

