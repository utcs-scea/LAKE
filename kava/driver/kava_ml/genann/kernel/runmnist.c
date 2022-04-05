#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <asm/segment.h>
#include <asm/uaccess.h>
#include <linux/buffer_head.h>

#include <asm/fpu/api.h>

#include "genann_kava.h"
#include "shared_memory.h"
#include "mnist.h"

#define description_string "Kernel implementation of MNIST in genann."
#define maintainer_string "Bodun Hu"

MODULE_AUTHOR(maintainer_string);
MODULE_DESCRIPTION(description_string);
MODULE_VERSION("0.01");
MODULE_LICENSE("GPL");  /* required for kernel_fpu_begin */

#define BUF_LEN 1024
static char imageFileName[BUF_LEN] __initdata;
module_param_string(imageFileName, imageFileName, BUF_LEN, S_IRUGO);

static char labelFileName[BUF_LEN] __initdata;
module_param_string(labelFileName, labelFileName, BUF_LEN, S_IRUGO);

//#define MEASURE_MICROBENCHMARKS
#define MEASURE_END2END_TIME

#define SIZE 28

static unsigned int mnistBinaryToInt(char *v) {
    int i;
    unsigned int result = 0;

    for (i = 0; i < 4; ++i) {
        result <<= 8;
        result |= (unsigned char)v[i];
    }

    return result;
}

struct file *file_open(const char *path, int flags, int rights) {
    struct file *filp = NULL;
    //mm_segment_t oldfs;
    int err = 0;
    //oldfs = get_fs();
    //set_fs(get_ds());
    filp = filp_open(path, flags, rights);
    //set_fs(oldfs);
    if (IS_ERR(filp)) {
        pr_err("file_open failed!\n");
        err = PTR_ERR(filp);
        return NULL;
    }
    return filp;
}

void file_close(struct file *file) {
    filp_close(file, NULL);
}

int loadMNISTDataUpToKern(const char *image_filename, const char *label_filename,
        mnist_data **data, unsigned int *count) {
    const int LABEL_MAGIC = 2049;
    const int IMAGE_MAGIC = 2051;

    const int ROWS = 28;
    const int COLUMNS = 28;

    int return_code = 0;
    int i;
    char temp[4];

    unsigned int image_count, label_count;
    unsigned int image_dimentions[2];

    // Open image and label files and check for validity
    struct file *imageFile = file_open(imageFileName, O_RDONLY, 0);
    struct file *labelFile = file_open(labelFileName, O_RDONLY, 0);
    if (!imageFile || !labelFile) {
        pr_err("Can't open image or label file.\n");
        return_code = -1;
    }
   
    loff_t img_pos = 0;
    loff_t label_pos = 0;
    int read_ret;
    read_ret = kernel_read(imageFile, (char *)temp, sizeof(int), &img_pos);
    if (mnistBinaryToInt(temp) != IMAGE_MAGIC) {
        pr_err("Bad magic number in image file! %d\n", mnistBinaryToInt(temp));
        return_code = -2;
    }
    //img_pos = 0;

    read_ret = kernel_read(labelFile, (char *)temp, sizeof(int), &label_pos);
    if (mnistBinaryToInt(temp) != LABEL_MAGIC) {
        pr_err("Bad magic number in label file! %d\n", mnistBinaryToInt(temp));
        return_code = -3;
    }
    //label_pos = 0;
    
    // Read image/label counts from the files and make sure there's the same
    // number in each file.
    read_ret = kernel_read(imageFile, (char *)temp, sizeof(int), &img_pos);
    image_count = mnistBinaryToInt(temp);

    read_ret = kernel_read(labelFile, (char *)temp, sizeof(int), &label_pos);
    label_count = mnistBinaryToInt(temp);

    if (image_count != label_count) {
        pr_err("The number of labels and images do not match!\n");
        return_code = -4;
    }

    // Get image dimensions from the file and make sure they match the expected
    // constant values.
    for (i = 0; i < 2; ++i) {
        read_ret = kernel_read(imageFile, (char *)temp, sizeof(int), &img_pos);
        image_dimentions[i] = mnistBinaryToInt(temp);
    }

    if (image_dimentions[0] != ROWS || image_dimentions[1] != COLUMNS) {
        pr_err("Image dimensions don't match the expected %dx%d", ROWS, COLUMNS);
        pr_err("Image dimensions don't match the expected %dx%d", image_dimentions[0], image_dimentions[1]);
        return_code = -2;
    }

    unsigned int *labelsArray = (unsigned int *)vmalloc(sizeof(unsigned int) * image_count);

    // count how many labels are less than LABELS_SIZE
    unsigned int countUpTo = 0;
    for (i = 0; i < image_count; ++i) {
        read_ret = kernel_read(labelFile, (char *)temp, 1, &label_pos);
        labelsArray[i] = temp[0];
        if (temp[0] < LABELS_SIZE) {
            ++countUpTo;
        }
    }

    // READ image/label data from the files and put in data array
    *count = countUpTo;
    *data = (mnist_data *)vmalloc(sizeof(mnist_data) * countUpTo);
    int j = 0;
    for (i = 0; i < image_count; ++i) {
        unsigned char read_data[ROWS * COLUMNS];
        read_ret = kernel_read(imageFile, (char *)read_data, ROWS * COLUMNS, &img_pos);

        if (labelsArray[i] >= LABELS_SIZE) continue;

        mnist_data *d = &(*data)[j++];
        memcpy(d->data, read_data, ROWS*COLUMNS);

        d->label = labelsArray[i];
    }
    if (imageFile) file_close(imageFile);
    if (labelFile) file_close(labelFile);
    
    return return_code;
}

void copy_to_double(mnist_data_ann *data_ann, mnist_data *data_raw, int cnt) {
    int i, j, k;
    for (i = 0; i < cnt; i++) {
        data_ann[i].label = (double)data_raw[i].label;
        for (j = 0; j < SIZE; j++) {
            for (k = 0; k < SIZE; k++) {
                data_ann[i].data[j*28+k] = (double)data_raw[i].data[j][k];
            }
        }
    }
}

static int __init genann_mnist_init(void) {
#ifdef MEASURE_MICROBENCHMARKS
    long total_init_micro = 0;
    struct timespec micro_init_start, micro_init_stop, micro_train_start,
                    micro_train_stop, micro_run_start, micro_run_stop,
                    micro_free_start, micro_free_stop;
    long total_run_micro = 0;
    long micro_nIter_run = 0;
    long total_train_micro = 0;
    long micro_nIter_train = 0;
    long total_free_micro = 0;
#endif

#ifdef MEASURE_END2END_TIME
    struct timespec micro_end2end_start, micro_end2end_stop;
    long total_end2end_micro = 0;
#endif

    int i, j, k;
    int correct = 0;
    double ** outs;
    mnist_data *data_raw;
    mnist_data_ann *data_ann;
    unsigned int cnt;
    int ret;

    pr_info("image file path: %s\n", imageFileName);
    pr_info("label file path: %s\n", labelFileName);

    printk(KERN_INFO "GENANN example 5.\n");
    printk(KERN_INFO "Train a MNIST on CUDA GPU.\n");

    //char* labelFileName = "mnist/t10k-labels.idx1-ubyte";
    //char* imageFileName = "mnist/t10k-images.idx3-ubyte";
    // TODO: take custom path

    if ((ret = loadMNISTDataUpToKern(imageFileName, labelFileName, &data_raw, &cnt))) {
        printk(KERN_INFO "An error occured: %d\n", ret);
    }

    data_ann = (mnist_data_ann*)kava_alloc(sizeof(mnist_data_ann) * cnt);
    if (data_ann == NULL) {
        pr_err("Failed to allocate shared memory buffer, size = %ld\n", sizeof(mnist_data_ann) * cnt);
        vfree(data_raw);
        return 0;
    }
    copy_to_double(data_ann, data_raw, cnt);

    printk(KERN_INFO "Loaded %d images.\n", cnt);

    outs = (double **)vmalloc(sizeof(double *) * cnt);

#ifdef MEASURE_END2END_TIME
    getnstimeofday(&micro_end2end_start); 
#endif

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_init_start);
#endif
    // New network with 2 inputs, 1 hidden layer of 2 neurons, and 1 output.
    genann *ann = genann_init(28*28, 3, 128, LABELS_SIZE);
#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_init_stop);
    total_init_micro += ELAPSED_TIME_MICRO_SEC(micro_init_start, micro_init_stop);
#endif
    double *arr = (double *)vmalloc(sizeof(double) * LABELS_SIZE);

    // Train on the four labeled data points many times.
    for (i = 0; i < 5; ++i) {
        for (j = 0; j < cnt; j++) {
            memset(arr, 0, sizeof(double) * LABELS_SIZE);
            arr[(int)data_ann[j].label] = 1;
#ifdef MEASURE_MICROBENCHMARKS
            getnstimeofday(&micro_train_start);
#endif
            genann_train(ann, data_ann[j].data, arr, 0.1);
#ifdef MEASURE_MICROBENCHMARKS
            getnstimeofday(&micro_train_stop);
            total_train_micro += ELAPSED_TIME_MICRO_SEC(micro_train_start, micro_train_stop);
            micro_nIter_train ++;
#endif
        }
    }

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_run_start);
#endif
    for (j = 0; j < cnt; j++) {
        outs[j] = genann_run(ann, data_ann[j].data);
    }
#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_run_stop);
    total_run_micro += ELAPSED_TIME_MICRO_SEC(micro_run_start, micro_run_stop);
    micro_nIter_run ++;
#endif

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_free_start);
#endif
    genann_free(ann);
#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_free_stop);
    total_free_micro += ELAPSED_TIME_MICRO_SEC(micro_free_start, micro_free_stop);
#endif

#ifdef MEASURE_END2END_TIME
    getnstimeofday(&micro_end2end_stop);
#endif

    kernel_fpu_begin();
    for (j = 0; j < cnt; j++) {
        int max = 0;
        for (k = 1; k < LABELS_SIZE; k++) {
            if (outs[j][k] > outs[j][max]) {
                max = k;
            }
            vfree(outs[j]);
        }
        if (max == (int)data_ann[j].label) correct++;
    }
    printk(KERN_INFO "Correct percentage: %d\n", (int)(1.0 * correct / cnt * 100));
    kernel_fpu_end();

#ifdef MEASURE_MICROBENCHMARKS
    PRINT(V_INFO, "Average genann init time: %ld usec\n", total_init_micro);
    PRINT(V_INFO, "Average genann train time: %ld usec\n", total_train_micro / micro_nIter_train);
    PRINT(V_INFO, "Average genann run time: %ld usec\n", total_run_micro / micro_nIter_run);
    PRINT(V_INFO, "Average genann free time: %ld usec\n", total_free_micro);
#endif
    vfree(data_raw);
    vfree(outs);
    kava_free(data_ann);
    vfree(arr);

#ifdef MEASURE_END2END_TIME
    total_end2end_micro = ELAPSED_TIME_MICRO_SEC(micro_end2end_start, micro_end2end_stop);
    PRINT(V_INFO, "Total execution time: %ld usec\n", total_end2end_micro);
#endif
    return 0;
}

static void __exit genann_mnist_exit(void) {
    printk(KERN_INFO "ANN is freed.\n");
}

module_init(genann_mnist_init);
module_exit(genann_mnist_exit);
