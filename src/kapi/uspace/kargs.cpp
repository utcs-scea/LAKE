#include <map>
#include <stdlib.h>
#include <string.h>
#include "kargs.h"

std::map<uint64_t, struct kernel_args_metadata*> kargs_metadata_map;

void init_kargs_kv() {
}   

struct kernel_args_metadata* get_kargs(const void* ptr) {
    uint64_t key = (uint64_t) ptr;
    auto it = kargs_metadata_map.find(key);

    if (it == kargs_metadata_map.end()) {
        struct kernel_args_metadata *metadata = new struct kernel_args_metadata();
        memset(metadata, 0, sizeof(struct kernel_args_metadata));
        kargs_metadata_map[key] = metadata;
        return metadata;
    }

    return it->second;
}

void destroy_kargs_kv()
{
    for (auto it = kargs_metadata_map.begin(); it != kargs_metadata_map.end(); it++) {
        delete it->second;
    }
}
