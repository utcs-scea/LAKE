#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/atomic.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/namei.h>

// Function prototypes
static int __init lake_fv_init(void);
static void __exit lake_fv_exit(void);

#define MAX_REGISTRIES 32
#define MAX_FVS 256

// TODO: handle list of fvs that wrap around

struct fv_metadata {
    struct timespec timestamp;
    char* fv;
};

struct registry {
    int id;
    char occupied;
    char* model_path;
    void* (*classifier)(void*);
    void* (*policy)(void*);
    // fvs
    spinlock_t fvs_spinlock;
    int window;
    // schema desc
    char* schema;
    int schema_offsets[MAX_FVS];
    int schema_sizes[MAX_FVS];
    int schema_sz;
    // feat vecs buffer
    struct fv_metadata fvs[MAX_FVS];
    void* buffer;
    atomic_t head;
    atomic_t tail;
};

struct registry registries[MAX_REGISTRIES];

void parse_schema(const char* input, int* offsets, int* sizes, int* sum) {
    if (input == NULL || offsets == NULL || sizes == NULL || sum == NULL) {
        return;
    }
    int length = strlen(input);
    int offset = 0;
    int count = 0;
    *sum = 0;
    int i;

    for (i = 0; i < length; ++i) {
        if (input[i] == 'c') {
            offsets[count++] = offset;
            offset += 1;
            *sum += 1;
            sizes[i] = 1;
        } else if (input[i] == 'i') {
            offsets[count++] = offset;
            offset += 4;
            *sum += 4;
            sizes[i] = 4;
        }
    }
}

int create_registry(char* schema, int window) {
    int i,j;
    for (i = 0; i < MAX_REGISTRIES; ++i) {
        if (!registries[i].occupied) {
            registries[i].occupied = 1;
            spin_lock_init(&registries[i].fvs_spinlock);
            registries[i].window = window;
            // set up schema
            registries[i].schema = kstrdup(schema, GFP_KERNEL);
            int count;
            parse_schema(schema, registries[i].schema_offsets, registries[i].schema_sizes, &count);
            registries[i].schema_sz = count;

            // set up fvs buffer
            atomic_set(&registries[i].head, 0);
            atomic_set(&registries[i].tail, 0);
            registries[i].buffer = kzalloc(MAX_FVS * registries[i].schema_sz, GFP_KERNEL);

            for (j = 0; j < MAX_FVS; ++j) {
                registries[i].fvs[j].fv = registries[i].buffer + (j * registries[i].schema_sz);
            }
            return i;
        }
    }
    return -ENOMEM;
}

void destroy_registry(int reg_id) {
    if (reg_id < 0 || reg_id >= MAX_REGISTRIES) {
        pr_err("error destroy_registry\n");
        return; // :)
    }
    kfree(registries[reg_id].schema_offsets);
    kfree(registries[reg_id].buffer);
    registries[reg_id].occupied = 0;
}

ssize_t write_file_data(const char *filename, const void *buf, size_t count)
{
    struct file *file;
    mm_segment_t old_fs;
    ssize_t ret = 0;

    // Open the file for writing
    file = filp_open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (IS_ERR(file)) {
        pr_err("Error opening file %s\n", filename);
        return PTR_ERR(file);
    }

    // Write data to the file
    old_fs = get_fs();
    set_fs(KERNEL_DS);
    ret = kernel_write(file, buf, count, &file->f_pos);
    set_fs(old_fs);
    filp_close(file, NULL);
    return ret;
}

ssize_t read_file_data(const char *filename, void *buf, size_t count, loff_t *pos)
{
    struct file *file;
    mm_segment_t old_fs;
    ssize_t ret = 0;
    file = filp_open(filename, O_RDONLY, 0);
    if (IS_ERR(file)) {
        pr_err("Error opening file %s\n", filename);
        return PTR_ERR(file);
    }

    // Read data from the file
    old_fs = get_fs();
    set_fs(KERNEL_DS);
    ret = kernel_read(file, buf, count, pos);
    set_fs(old_fs);
    filp_close(file, NULL);

    return ret;
}

int delete_file(const char *filename)
{
    struct path path;
    int ret;
    ret = kern_path_create(AT_FDCWD, filename, &path, 0);
    if (ret) {
        pr_err("Error resolving path for file %s\n", filename);
        return ret;
    }
    ret = vfs_unlink(path.dentry->d_parent->d_inode, path.dentry, NULL);
    if (ret) {
        pr_err("Error unlinking (deleting) file %s\n", filename);
    }
    path_put(&path);
    return ret;
}

void create_model(int reg_id, char* filepath) {
    char flag = 1;
    write_file_data(filepath, &flag, 1);
    registries[reg_id].model_path = kstrdup(filepath, GFP_KERNEL);
}

void update_model(int reg_id, char* model, uint32_t model_size) {
    write_file_data(registries[reg_id].model_path, model, model_size);    
}

void load_model(int reg_id, char* model, uint32_t model_size) {
    read_file_data(registries[reg_id].model_path, model, model_size, 0);
}

void delete_model(int reg_id, char* filepath) {
    delete_file(registries[reg_id].model_path);
}

void register_classifier(int reg_id, void* (*fn)(void*)) {
    registries[reg_id].classifier = fn;
}

void register_policy(int reg_id, void* (*fn)(void*)) {
    registries[reg_id].policy = fn;
}

struct fv_metadata* find_first_entry_after_timestamp(struct registry* reg, struct timespec ts, int* count) {
    int tail, head;
    size_t elements_after_ts = 0;

    tail = atomic_read(&reg->tail);
    head = atomic_read(&reg->head);

    while (tail != head) {
        struct fv_metadata* entry = &reg->fvs[tail];
        if (timespec_compare(&entry->timestamp, &ts) > 0) {
            // Found an entry with timestamp higher than TS
            *count = elements_after_ts;
            return entry;
        }
        // Move to the next entry
        tail = (tail + 1) % (MAX_FVS * reg->schema_sz);
        elements_after_ts++;
    }

    // No entry with timestamp higher than TS found
    *count = elements_after_ts;
    return NULL;
}

void* score_features(int reg_id, struct timespec after, int batch_size) {
    int count;
    struct fv_metadata* first = find_first_entry_after_timestamp(&registries[reg_id], after, &count);

    // get min to window
    count = count > registries[reg_id].window ? registries[reg_id].window : count;

    char* fvs_start = first->fv;
    // start of array of fvs and # fvs and batch
    void* args[] = {fvs_start, (void*)count, (void*)batch_size};

    // TODO: handle list of fvs that wrap around

    // there could be a race condition here, but this depends
    // on the characteristics of the workload running
    return registries[reg_id].classifier(args);
}

// returns a pointer to all fvs, consecutively
char* get_features(int reg_id, struct timespec after, int* count) {
    // how consistent do you want the view?
    // if it needs to be super, we need to either lock the entire
    // fv cbuf or make a copy
    struct fv_metadata* first = find_first_entry_after_timestamp(&registries[reg_id], after, count);
    char* fvs_start = first->fv;
    return fvs_start;
}


void remove_entries_before_timestamp(struct registry* reg, struct timespec ts) {
    int tail, head;
    tail = atomic_read(&reg->tail);
    head = atomic_read(&reg->head);

    while (tail != head) {
        struct fv_metadata* entry = &reg->fvs[tail];
        if (timespec_compare(&entry->timestamp, &ts) < 0) {
            // Move to the next entry
            tail = (tail + 1) % (MAX_FVS * reg->schema_sz);
        } else {
            break;
        }
    }
    // Update the tail index
    atomic_set(&reg->tail, tail);
}

void truncate_features(int reg_id, struct timespec after) {
    remove_entries_before_timestamp(&registries[reg_id], after);
}

int get_empty_entry(struct registry* reg) {
    int head, tail, next;
    tail = atomic_read(&reg->tail);
    head = atomic_read(&reg->head);
    next = (head + 1) % MAX_FVS;
    // not full
    if (next != tail)
        return head;
    else
        return -1;
}

int begin_fv_capture(int reg_id) {
    int idx = get_empty_entry(&registries[reg_id]);
    if (idx == -1)
        return -ENOMEM; // full
    atomic_set(&registries[reg_id].head, (atomic_read(&registries[reg_id].head) + 1) % MAX_FVS);
    return idx;
}

// fv_id comes from the return of begin_fv_capture
// depending on the workload, this can be changed to use a
// thread local value, instead of an int
void capture_feature(int reg_id, int fv_id, char feat_id, char* data) {
    char* fv = registries[reg_id].fvs[fv_id].fv;
    fv = fv + registries[reg_id].schema_offsets[feat_id];
    memcpy(fv, data, registries[reg_id].schema_sizes[feat_id]);
}

void capture_feature_incr(int reg_id, int fv_id, char feat_id, int add) {
    char* fv = registries[reg_id].fvs[fv_id].fv;
    fv = fv + registries[reg_id].schema_offsets[feat_id];

    int size = registries[reg_id].schema_sizes[feat_id];
    if (size == 1) {
        char* p = (char*) fv;
        (*p) += add;
    } else if (size == 4) {
        int* p = (int*) fv;
        (*p) += add;
    }
}

void commit_fv_capture(int reg_id, int sys_id) {
    // pass for now since we are using indexes, not thread local
}

static int __init lake_fv_init(void) {
    // TODO: kernel trampolines

    printk(KERN_INFO "lake_fv: module loaded\n");
    return 0;
}

static void __exit lake_fv_exit(void) {
    // TODO: cleanup
    printk(KERN_INFO "lake_fv: module unloaded\n");
}

// Register the module initialization and exit functions
module_init(lake_fv_init);
module_exit(lake_fv_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("My Kernel Module");
MODULE_AUTHOR("Your Name");