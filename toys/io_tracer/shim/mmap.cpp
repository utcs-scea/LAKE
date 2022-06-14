#include "common.h"
#include "mmap.h"

#include "reads.h"

std::mutex init_mtx;
uint8_t uff_initd(0);

static mmap_ptr real_mmap = 0;

void mmap_contructor() {
    real_mmap = (mmap_ptr) dlsym(RTLD_NEXT, "mmap");
    if (real_mmap == NULL) {
        printf("Failed getting original mmap\n");
        exit(1);
    }
}

std::mutex list_mtx;
//performance doesnt matter, so use a simple list
static std::list<file_mapping> mappings;

/*
 *  userfaultfd code
 */

static int uffd;
static pthread_t uffd_thread;
char file_path[512];
char proc_path[512];

static bool belongs_to_mapping(file_mapping &m, uint64_t address) {
    if (address >= m.start && address < m.end)
        return true;
    return false;
}

void handle_fault(uint64_t address, char* buf) {
    for (file_mapping &m : mappings) {
        if (belongs_to_mapping(m, address)) {
            // we found the mapping this address is from, read from file

            //int inode = inode_from_fd(m.fd);

            // sprintf(proc_path, "/proc/self/fd/%d", m.fd);
            // uint64_t n = readlink(proc_path, file_path, 511);
            // file_path[n] = 0;
            //fprintf(stderr, "%s, %d, %u\n", file_path, inode, page_offset);

            uint64_t foffset = address - m.start;
            uint32_t page_offset = foffset / PAGE_SZ;
            //fprintf(stderr, ":m,%d,%u\n", inode, page_offset);
            fprintf(stderr, ":m,%d,%u\n", m.fd, page_offset);

            // go to page aligned position of file, read, reset
            uint32_t old_pos = lseek(m.fd, 0, SEEK_CUR);
            lseek(m.fd, page_offset*PAGE_SZ, SEEK_SET);
            uint64_t nread = real_read_225(m.fd, buf, 4096);
            lseek(m.fd, old_pos, SEEK_SET);
        }
	}
}

static void *handler(void *arg)
{
    long page_size = 4096;
    char buf[page_size];

    for (;;) {
        struct uffd_msg msg;

        struct pollfd pollfd[1];
        pollfd[0].fd = uffd;
        pollfd[0].events = POLLIN;

        // wait for a userfaultfd event to occur
        int pollres = poll(pollfd, 1, 2000);

        //if (stop)
        //    return NULL;

        switch (pollres) {
        case -1:
            perror("poll/userfaultfd");
            continue;
        case 0:
            continue;
        case 1:
            break;
        default:
            fprintf(stderr, "unexpected poll result\n");
            exit(1);
        }

        //printf("received uff\n");

        if (pollfd[0].revents & POLLERR) {
            fprintf(stderr, "pollerr\n");
            exit(1);
        }

        if (!pollfd[0].revents & POLLIN) {
            continue;
        }

        int readres = read(uffd, &msg, sizeof(msg));
        if (readres == -1) {
            if (errno == EAGAIN)
                continue;
            perror("read/userfaultfd");
            exit(1);
        }

        if (readres != sizeof(msg)) {
            fprintf(stderr, "invalid msg size\n");
            exit(1);
        }

        if (msg.event & UFFD_EVENT_PAGEFAULT) {
            long long addr = msg.arg.pagefault.address;
            printf("handling page fault at %p\n", addr);
            
            handle_fault(addr, buf);
            //set destination to be page aligned
            addr = addr & ~(PAGE_SZ - 1);

            struct uffdio_copy copy;
            copy.src = (long long)buf;
            copy.dst = (long long)addr;
            copy.len = page_size;
            copy.mode = 0;
            if (ioctl(uffd, UFFDIO_COPY, &copy) == -1) {
                perror("ioctl/copy");
                exit(1);
            }
        }
    }

    return NULL;
}

void start_uff_thread(void) {
    // open the userfault fd
    uffd = syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK);
    if (uffd == -1) {
        perror("> syscall/userfaultfd");
        exit(1);
    }

    // enable for api version and check features
    struct uffdio_api uffdio_api;
    uffdio_api.api = UFFD_API;
    uffdio_api.features = 0;
    if (ioctl(uffd, UFFDIO_API, &uffdio_api) == -1) {
        perror("ioctl/uffdio_api");
        exit(1);
    }

    if (uffdio_api.api != UFFD_API) {
        fprintf(stderr, "unsupported userfaultfd api\n");
        exit(1);
    }

    printf("UFF registered, starting UFF thread\n");
    pthread_create(&uffd_thread, NULL, handler, NULL);
    sleep(0.1);
}

void check_uff_initd(void) {
    //make sure we only init once
    init_mtx.lock();
    if (uff_initd == 0) {
        uff_initd = 1;
        start_uff_thread();
    }
    init_mtx.unlock();
}


uint64_t register_uff_area(int fd, size_t length, int prot, int flags) {
    // allocate a memory region to be managed by userfaultfd
    void* region = real_mmap(NULL, length, prot, flags|MAP_ANONYMOUS, -1, 0);
    //void* region = real_mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (region == MAP_FAILED) {
        perror("real_mmap");
        return 0;
    }

    // register the pages in the region for missing callbacks
    struct uffdio_register uffdio_register;
    uffdio_register.range.start = (unsigned long)region;
    uffdio_register.range.len = length;
    uffdio_register.mode = UFFDIO_REGISTER_MODE_MISSING;
    if (ioctl(uffd, UFFDIO_REGISTER, &uffdio_register) == -1) {
        perror("ioctl/uffdio_register");
        return 0;
    }

    if ((uffdio_register.ioctls & UFFD_API_RANGE_IOCTLS) !=
            UFFD_API_RANGE_IOCTLS) {
        fprintf(stderr, "unexpected userfaultfd ioctl set\n");
        return 0;
    }

    file_mapping fm;
    fm.start = (uint64_t) region;
    fm.end = ((uint64_t) region) + length;
    fm.fd = fd;
    mappings.push_back(fm);

    return (uint64_t) region;
}

//serialize mmaps for now
std::mutex mmap_mtx;

// void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
//     mmap_mtx.lock();

//     //printf("In mmap shim\n");
//     check_uff_initd();

//     // ignore anon mappings
//     if (flags & MAP_ANONYMOUS) {
//         printf("Ignoring anon mapping\n");
//         void* ret = real_mmap(addr, length, prot, flags, fd, offset);
//         mmap_mtx.unlock();
//         return ret;
//     } 

//     //TODO: deal with offset
//     printf("Caught file-backed mapping\n");
    
//     printf("fd: %d  offset %lu\n", fd, offset);
//     //void* ret = real_mmap(addr, length, prot, flags, fd, offset);
//     //mmap_mtx.unlock();
//     //return ret;

//     uint64_t region = register_uff_area(fd, length, prot, flags);
//     if (region == 0) {
//         printf("Error mapping UFF area\n");
//         exit(1);
//     }

//     mmap_mtx.unlock();
//     return (void*) region;   
// }