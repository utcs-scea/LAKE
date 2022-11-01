/*
 * Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
 * Copyright (C) 2022-2024 Henrique Fingler
 * Copyright (C) 2022-2024 Isha Tarte
 * Copyright (C) 2022-2024 Hangchen Yu
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */



#include <linux/types.h>
#include <linux/rhashtable.h>
#include <linux/vmalloc.h>
#include "kargs.h"

struct rhashtable kargs_metadata_map;
DEFINE_MUTEX(kargs_metadata_map_mutex);

struct metadata_object {
    const void *key;
    struct rhash_head linkage;
    struct kernel_args_metadata *metadata;
};

const static struct rhashtable_params metadata_object_params = {
    .key_len     = sizeof(const void *),
    .key_offset  = offsetof(struct metadata_object, key),
    .head_offset = offsetof(struct metadata_object, linkage),
};

void init_kargs_kv(void) {
    int r = rhashtable_init(&kargs_metadata_map, &metadata_object_params);
    if (r) {
        pr_err("Error on rhashtable_init\n");
    }
}

struct kernel_args_metadata* get_kargs(const void* ptr) {
    struct metadata_object *object;
    struct kernel_args_metadata *metadata;

    mutex_lock(&kargs_metadata_map_mutex);

    object = rhashtable_lookup_fast(&kargs_metadata_map, &ptr, metadata_object_params);
    if (object == NULL) {
        metadata = (struct kernel_args_metadata *) vmalloc(sizeof(struct kernel_args_metadata));
        memset(metadata, 0, sizeof(struct kernel_args_metadata));
        object = (struct metadata_object *) vmalloc(sizeof(struct metadata_object));
        object->key = ptr;
        object->metadata = metadata;
        rhashtable_insert_fast(&kargs_metadata_map, &object->linkage, metadata_object_params);
    }

    mutex_unlock(&kargs_metadata_map_mutex);
    return object->metadata;
}

static void metadata_map_free_fn(void *ptr, void *arg)
{
    struct metadata_object *obj = (struct metadata_object *)ptr;
    if (obj) {
        vfree(obj->metadata);
        vfree(obj);
    }
}

void destroy_kargs_kv(void)
{
    rhashtable_free_and_destroy(&kargs_metadata_map, metadata_map_free_fn, NULL);
}
