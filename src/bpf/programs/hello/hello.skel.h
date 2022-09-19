/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */

/* THIS FILE IS AUTOGENERATED! */
#ifndef __HELLO_BPF_SKEL_H__
#define __HELLO_BPF_SKEL_H__

#include <errno.h>
#include <stdlib.h>
#include <bpf/libbpf.h>

struct hello_bpf {
	struct bpf_object_skeleton *skeleton;
	struct bpf_object *obj;
	struct {
		struct bpf_map *rodata;
		struct bpf_map *bss;
	} maps;
	struct {
		struct bpf_program *handle_tp;
	} progs;
	struct {
		struct bpf_link *handle_tp;
	} links;
	struct hello_bpf__bss {
		int my_pid;
	} *bss;
	struct hello_bpf__rodata {
	} *rodata;
};

static void
hello_bpf__destroy(struct hello_bpf *obj)
{
	if (!obj)
		return;
	if (obj->skeleton)
		bpf_object__destroy_skeleton(obj->skeleton);
	free(obj);
}

static inline int
hello_bpf__create_skeleton(struct hello_bpf *obj);

static inline struct hello_bpf *
hello_bpf__open_opts(const struct bpf_object_open_opts *opts)
{
	struct hello_bpf *obj;
	int err;

	obj = (struct hello_bpf *)calloc(1, sizeof(*obj));
	if (!obj) {
		errno = ENOMEM;
		return NULL;
	}

	err = hello_bpf__create_skeleton(obj);
	err = err ?: bpf_object__open_skeleton(obj->skeleton, opts);
	if (err)
		goto err_out;

	return obj;
err_out:
	hello_bpf__destroy(obj);
	errno = -err;
	return NULL;
}

static inline struct hello_bpf *
hello_bpf__open(void)
{
	return hello_bpf__open_opts(NULL);
}

static inline int
hello_bpf__load(struct hello_bpf *obj)
{
	return bpf_object__load_skeleton(obj->skeleton);
}

static inline struct hello_bpf *
hello_bpf__open_and_load(void)
{
	struct hello_bpf *obj;
	int err;

	obj = hello_bpf__open();
	if (!obj)
		return NULL;
	err = hello_bpf__load(obj);
	if (err) {
		hello_bpf__destroy(obj);
		errno = -err;
		return NULL;
	}
	return obj;
}

static inline int
hello_bpf__attach(struct hello_bpf *obj)
{
	return bpf_object__attach_skeleton(obj->skeleton);
}

static inline void
hello_bpf__detach(struct hello_bpf *obj)
{
	return bpf_object__detach_skeleton(obj->skeleton);
}

static inline int
hello_bpf__create_skeleton(struct hello_bpf *obj)
{
	struct bpf_object_skeleton *s;

	s = (struct bpf_object_skeleton *)calloc(1, sizeof(*s));
	if (!s)
		goto err;

	s->sz = sizeof(*s);
	s->name = "hello_bpf";
	s->obj = &obj->obj;

	/* maps */
	s->map_cnt = 2;
	s->map_skel_sz = sizeof(*s->maps);
	s->maps = (struct bpf_map_skeleton *)calloc(s->map_cnt, s->map_skel_sz);
	if (!s->maps)
		goto err;

	s->maps[0].name = "hello_bp.rodata";
	s->maps[0].map = &obj->maps.rodata;
	s->maps[0].mmaped = (void **)&obj->rodata;

	s->maps[1].name = "hello_bp.bss";
	s->maps[1].map = &obj->maps.bss;
	s->maps[1].mmaped = (void **)&obj->bss;

	/* programs */
	s->prog_cnt = 1;
	s->prog_skel_sz = sizeof(*s->progs);
	s->progs = (struct bpf_prog_skeleton *)calloc(s->prog_cnt, s->prog_skel_sz);
	if (!s->progs)
		goto err;

	s->progs[0].name = "handle_tp";
	s->progs[0].prog = &obj->progs.handle_tp;
	s->progs[0].link = &obj->links.handle_tp;

	s->data_sz = 5256;
	s->data = (void *)"\
\x7f\x45\x4c\x46\x02\x01\x01\0\0\0\0\0\0\0\0\0\x01\0\xf7\0\x01\0\0\0\0\0\0\0\0\
\0\0\0\0\0\0\0\0\0\0\0\xc8\x0e\0\0\0\0\0\0\0\0\0\0\x40\0\0\0\0\0\x40\0\x17\0\
\x01\0\x85\0\0\0\x0e\0\0\0\x77\0\0\0\x20\0\0\0\x18\x01\0\0\0\0\0\0\0\0\0\0\0\0\
\0\0\x61\x11\0\0\0\0\0\0\x5d\x01\x05\0\0\0\0\0\x18\x01\0\0\0\0\0\0\0\0\0\0\0\0\
\0\0\xb7\x02\0\0\x1c\0\0\0\xbf\x03\0\0\0\0\0\0\x85\0\0\0\x06\0\0\0\xb7\0\0\0\0\
\0\0\0\x95\0\0\0\0\0\0\0\x44\x75\x61\x6c\x20\x42\x53\x44\x2f\x47\x50\x4c\0\0\0\
\0\x42\x50\x46\x20\x74\x72\x69\x67\x67\x65\x72\x65\x64\x20\x66\x72\x6f\x6d\x20\
\x50\x49\x44\x20\x25\x64\x2e\x0a\0\x10\0\0\0\0\0\0\0\x58\0\0\0\0\0\0\0\x01\0\
\x50\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x01\x11\x01\x25\x0e\x13\x05\x03\x0e\x10\
\x17\x1b\x0e\x11\x01\x12\x06\0\0\x02\x34\0\x03\x0e\x49\x13\x3f\x19\x3a\x0b\x3b\
\x0b\x02\x18\0\0\x03\x01\x01\x49\x13\0\0\x04\x21\0\x49\x13\x37\x0b\0\0\x05\x24\
\0\x03\x0e\x3e\x0b\x0b\x0b\0\0\x06\x24\0\x03\x0e\x0b\x0b\x3e\x0b\0\0\x07\x2e\
\x01\x11\x01\x12\x06\x40\x18\x97\x42\x19\x03\x0e\x3a\x0b\x3b\x0b\x27\x19\x49\
\x13\x3f\x19\0\0\x08\x34\0\x03\x0e\x49\x13\x3a\x0b\x3b\x0b\x02\x18\0\0\x09\x05\
\0\x03\x0e\x3a\x0b\x3b\x0b\x49\x13\0\0\x0a\x34\0\x02\x17\x03\x0e\x3a\x0b\x3b\
\x0b\x49\x13\0\0\x0b\x26\0\x49\x13\0\0\x0c\x34\0\x03\x0e\x49\x13\x3a\x0b\x3b\
\x05\0\0\x0d\x0f\0\x49\x13\0\0\x0e\x15\0\x49\x13\x27\x19\0\0\x0f\x16\0\x49\x13\
\x03\x0e\x3a\x0b\x3b\x0b\0\0\x10\x34\0\x03\x0e\x49\x13\x3a\x0b\x3b\x0b\0\0\x11\
\x15\x01\x49\x13\x27\x19\0\0\x12\x05\0\x49\x13\0\0\x13\x18\0\0\0\x14\x0f\0\0\0\
\0\x34\x01\0\0\x04\0\0\0\0\0\x08\x01\0\0\0\0\x0c\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\
\0\0\0\0\0\0\x68\0\0\0\x02\0\0\0\0\x3f\0\0\0\x01\x06\x09\x03\0\0\0\0\0\0\0\0\
\x03\x4b\0\0\0\x04\x52\0\0\0\x0d\0\x05\0\0\0\0\x06\x01\x06\0\0\0\0\x08\x07\x02\
\0\0\0\0\x6e\0\0\0\x01\x08\x09\x03\0\0\0\0\0\0\0\0\x05\0\0\0\0\x05\x04\x07\0\0\
\0\0\0\0\0\0\x68\0\0\0\x01\x5a\0\0\0\0\x01\x0b\x6e\0\0\0\x08\0\0\0\0\xbe\0\0\0\
\x01\x12\x09\x03\0\0\0\0\0\0\0\0\x09\0\0\0\0\x01\x0b\x36\x01\0\0\x0a\0\0\0\0\0\
\0\0\0\x01\x0d\x6e\0\0\0\0\x03\xca\0\0\0\x04\x52\0\0\0\x1c\0\x0b\x4b\0\0\0\x0c\
\0\0\0\0\xdb\0\0\0\x03\x6f\x01\x0d\xe0\0\0\0\x0e\xe5\0\0\0\x0f\xf0\0\0\0\0\0\0\
\0\x02\x1f\x05\0\0\0\0\x07\x08\x10\0\0\0\0\x02\x01\0\0\x03\xb0\x0d\x07\x01\0\0\
\x11\x18\x01\0\0\x12\x1f\x01\0\0\x12\x24\x01\0\0\x13\0\x05\0\0\0\0\x05\x08\x0d\
\xca\0\0\0\x0f\x2f\x01\0\0\0\0\0\0\x02\x1b\x05\0\0\0\0\x07\x04\x14\0\x55\x62\
\x75\x6e\x74\x75\x20\x63\x6c\x61\x6e\x67\x20\x76\x65\x72\x73\x69\x6f\x6e\x20\
\x31\x31\x2e\x31\x2e\x30\x2d\x2b\x2b\x32\x30\x32\x31\x31\x30\x31\x31\x30\x39\
\x34\x31\x35\x39\x2b\x31\x66\x64\x65\x63\x35\x39\x62\x66\x66\x63\x31\x2d\x31\
\x7e\x65\x78\x70\x31\x7e\x32\x30\x32\x31\x31\x30\x31\x31\x32\x31\x34\x36\x31\
\x34\x2e\x38\0\x68\x65\x6c\x6c\x6f\x2e\x62\x70\x66\x2e\x63\0\x2f\x68\x6f\x6d\
\x65\x2f\x68\x66\x69\x6e\x67\x6c\x65\x72\x2f\x68\x66\x2d\x48\x41\x43\x4b\x2f\
\x73\x72\x63\x2f\x74\x65\x73\x74\x5f\x62\x63\x63\x5f\x6b\x66\x75\x6e\x63\x2f\
\x62\x70\x66\x2f\x68\x65\x6c\x6c\x6f\0\x4c\x49\x43\x45\x4e\x53\x45\0\x63\x68\
\x61\x72\0\x5f\x5f\x41\x52\x52\x41\x59\x5f\x53\x49\x5a\x45\x5f\x54\x59\x50\x45\
\x5f\x5f\0\x6d\x79\x5f\x70\x69\x64\0\x69\x6e\x74\0\x5f\x5f\x5f\x5f\x66\x6d\x74\
\0\x62\x70\x66\x5f\x67\x65\x74\x5f\x63\x75\x72\x72\x65\x6e\x74\x5f\x70\x69\x64\
\x5f\x74\x67\x69\x64\0\x6c\x6f\x6e\x67\x20\x6c\x6f\x6e\x67\x20\x75\x6e\x73\x69\
\x67\x6e\x65\x64\x20\x69\x6e\x74\0\x5f\x5f\x75\x36\x34\0\x62\x70\x66\x5f\x74\
\x72\x61\x63\x65\x5f\x70\x72\x69\x6e\x74\x6b\0\x6c\x6f\x6e\x67\x20\x69\x6e\x74\
\0\x75\x6e\x73\x69\x67\x6e\x65\x64\x20\x69\x6e\x74\0\x5f\x5f\x75\x33\x32\0\x68\
\x61\x6e\x64\x6c\x65\x5f\x74\x70\0\x63\x74\x78\0\x70\x69\x64\0\x9f\xeb\x01\0\
\x18\0\0\0\0\0\0\0\x10\x01\0\0\x10\x01\0\0\x31\x01\0\0\0\0\0\0\0\0\0\x02\0\0\0\
\0\0\0\0\0\x01\0\0\x0d\x03\0\0\0\x01\0\0\0\x01\0\0\0\x05\0\0\0\0\0\0\x01\x04\0\
\0\0\x20\0\0\x01\x09\0\0\0\x01\0\0\x0c\x02\0\0\0\xe2\0\0\0\0\0\0\x01\x01\0\0\0\
\x08\0\0\x01\0\0\0\0\0\0\0\x03\0\0\0\0\x05\0\0\0\x07\0\0\0\x0d\0\0\0\xe7\0\0\0\
\0\0\0\x01\x04\0\0\0\x20\0\0\0\xfb\0\0\0\0\0\0\x0e\x06\0\0\0\x01\0\0\0\x03\x01\
\0\0\0\0\0\x0e\x03\0\0\0\x01\0\0\0\0\0\0\0\0\0\0\x0a\x05\0\0\0\0\0\0\0\0\0\0\
\x03\0\0\0\0\x0a\0\0\0\x07\0\0\0\x1c\0\0\0\x0a\x01\0\0\0\0\0\x0e\x0b\0\0\0\0\0\
\0\0\x1c\x01\0\0\x01\0\0\x0f\0\0\0\0\x09\0\0\0\0\0\0\0\x04\0\0\0\x21\x01\0\0\
\x01\0\0\x0f\0\0\0\0\x0c\0\0\0\0\0\0\0\x1c\0\0\0\x29\x01\0\0\x01\0\0\x0f\0\0\0\
\0\x08\0\0\0\0\0\0\0\x0d\0\0\0\0\x63\x74\x78\0\x69\x6e\x74\0\x68\x61\x6e\x64\
\x6c\x65\x5f\x74\x70\0\x74\x70\x2f\x72\x61\x77\x5f\x73\x79\x73\x63\x61\x6c\x6c\
\x73\x2f\x73\x79\x73\x5f\x65\x6e\x74\x65\x72\0\x2f\x68\x6f\x6d\x65\x2f\x68\x66\
\x69\x6e\x67\x6c\x65\x72\x2f\x68\x66\x2d\x48\x41\x43\x4b\x2f\x73\x72\x63\x2f\
\x74\x65\x73\x74\x5f\x62\x63\x63\x5f\x6b\x66\x75\x6e\x63\x2f\x62\x70\x66\x2f\
\x68\x65\x6c\x6c\x6f\x2f\x68\x65\x6c\x6c\x6f\x2e\x62\x70\x66\x2e\x63\0\x09\x69\
\x6e\x74\x20\x70\x69\x64\x20\x3d\x20\x62\x70\x66\x5f\x67\x65\x74\x5f\x63\x75\
\x72\x72\x65\x6e\x74\x5f\x70\x69\x64\x5f\x74\x67\x69\x64\x28\x29\x20\x3e\x3e\
\x20\x33\x32\x3b\0\x09\x69\x66\x20\x28\x70\x69\x64\x20\x21\x3d\x20\x6d\x79\x5f\
\x70\x69\x64\x29\0\x09\x62\x70\x66\x5f\x70\x72\x69\x6e\x74\x6b\x28\x22\x42\x50\
\x46\x20\x74\x72\x69\x67\x67\x65\x72\x65\x64\x20\x66\x72\x6f\x6d\x20\x50\x49\
\x44\x20\x25\x64\x2e\x5c\x6e\x22\x2c\x20\x70\x69\x64\x29\x3b\0\x7d\0\x63\x68\
\x61\x72\0\x5f\x5f\x41\x52\x52\x41\x59\x5f\x53\x49\x5a\x45\x5f\x54\x59\x50\x45\
\x5f\x5f\0\x4c\x49\x43\x45\x4e\x53\x45\0\x6d\x79\x5f\x70\x69\x64\0\x68\x61\x6e\
\x64\x6c\x65\x5f\x74\x70\x2e\x5f\x5f\x5f\x5f\x66\x6d\x74\0\x2e\x62\x73\x73\0\
\x2e\x72\x6f\x64\x61\x74\x61\0\x6c\x69\x63\x65\x6e\x73\x65\0\x9f\xeb\x01\0\x20\
\0\0\0\0\0\0\0\x14\0\0\0\x14\0\0\0\x6c\0\0\0\x80\0\0\0\0\0\0\0\x08\0\0\0\x13\0\
\0\0\x01\0\0\0\0\0\0\0\x04\0\0\0\x10\0\0\0\x13\0\0\0\x06\0\0\0\0\0\0\0\x2d\0\0\
\0\x6d\0\0\0\x0c\x34\0\0\x08\0\0\0\x2d\0\0\0\x6d\0\0\0\x27\x34\0\0\x10\0\0\0\
\x2d\0\0\0\x9a\0\0\0\x0d\x3c\0\0\x28\0\0\0\x2d\0\0\0\x9a\0\0\0\x06\x3c\0\0\x30\
\0\0\0\x2d\0\0\0\xae\0\0\0\x02\x48\0\0\x58\0\0\0\x2d\0\0\0\xe0\0\0\0\x01\x54\0\
\0\0\0\0\0\0\x0c\0\0\0\xff\xff\xff\xff\x04\0\x08\0\x08\x7c\x0b\0\x14\0\0\0\0\0\
\0\0\0\0\0\0\0\0\0\0\x68\0\0\0\0\0\0\0\xa0\0\0\0\x04\0\x70\0\0\0\x08\x01\x01\
\xfb\x0e\x0d\0\x01\x01\x01\x01\0\0\0\x01\0\0\x01\x2f\x75\x73\x72\x2f\x69\x6e\
\x63\x6c\x75\x64\x65\x2f\x61\x73\x6d\x2d\x67\x65\x6e\x65\x72\x69\x63\0\x2f\x75\
\x73\x72\x2f\x69\x6e\x63\x6c\x75\x64\x65\x2f\x62\x70\x66\0\0\x68\x65\x6c\x6c\
\x6f\x2e\x62\x70\x66\x2e\x63\0\0\0\0\x69\x6e\x74\x2d\x6c\x6c\x36\x34\x2e\x68\0\
\x01\0\0\x62\x70\x66\x5f\x68\x65\x6c\x70\x65\x72\x5f\x64\x65\x66\x73\x2e\x68\0\
\x02\0\0\0\0\x09\x02\0\0\0\0\0\0\0\0\x03\x0b\x01\x05\x0c\x0a\x13\x05\x27\x06\
\x20\x05\x0d\x06\x22\x05\x06\x06\x3c\x05\x02\x06\x23\x05\x01\x5b\x02\x02\0\x01\
\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\xc5\0\0\0\x04\0\
\xf1\xff\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x0c\0\0\0\0\0\0\0\0\0\0\0\
\0\0\0\0\0\0\0\0\0\0\0\0\x0c\0\x52\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\
\x0c\0\x5e\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x0c\0\x92\0\0\0\0\0\0\0\0\
\0\0\0\0\0\0\0\0\0\0\0\0\0\x0c\0\x9a\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\
\x0c\0\x9f\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x0c\0\xb3\0\0\0\0\0\0\0\0\
\0\0\0\0\0\0\0\0\0\0\0\0\0\x0c\0\xba\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\
\x0c\0\x29\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x0c\0\xbe\0\0\0\0\0\0\0\
\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x0c\0\x33\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\
\0\0\x0c\0\x37\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x0c\0\xc6\0\0\0\0\0\
\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x0c\0\xf6\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\
\0\0\0\x0c\0\xdf\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x0c\0\xfc\0\0\0\0\0\
\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x0c\0\x0d\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\
\0\0\0\0\x0c\0\x23\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x0c\0\x16\x01\0\
\0\0\0\0\0\0\0\0\0\0\0\0\0\xfa\0\0\0\0\0\x03\0\x58\0\0\0\0\0\0\0\0\0\0\0\0\0\0\
\0\x22\0\0\0\x01\0\x07\0\0\0\0\0\0\0\0\0\x1c\0\0\0\0\0\0\0\0\0\0\0\x03\0\x03\0\
\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x03\0\x07\0\0\0\0\0\0\0\0\0\0\0\0\0\0\
\0\0\0\0\0\0\0\x03\0\x08\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x03\0\x09\0\
\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x03\0\x11\0\0\0\0\0\0\0\0\0\0\0\0\0\0\
\0\0\0\0\0\0\0\x03\0\x13\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\xf2\0\0\0\x11\0\x05\
\0\0\0\0\0\0\0\0\0\x0d\0\0\0\0\0\0\0\x62\0\0\0\x12\0\x03\0\0\0\0\0\0\0\0\0\x68\
\0\0\0\0\0\0\0\xb3\0\0\0\x11\0\x06\0\0\0\0\0\0\0\0\0\x04\0\0\0\0\0\0\0\x10\0\0\
\0\0\0\0\0\x01\0\0\0\x1f\0\0\0\x30\0\0\0\0\0\0\0\x01\0\0\0\x18\0\0\0\x06\0\0\0\
\0\0\0\0\x0a\0\0\0\x1a\0\0\0\x0c\0\0\0\0\0\0\0\x0a\0\0\0\x02\0\0\0\x12\0\0\0\0\
\0\0\0\x0a\0\0\0\x03\0\0\0\x16\0\0\0\0\0\0\0\x0a\0\0\0\x1c\0\0\0\x1a\0\0\0\0\0\
\0\0\x0a\0\0\0\x04\0\0\0\x1e\0\0\0\0\0\0\0\x01\0\0\0\x17\0\0\0\x2b\0\0\0\0\0\0\
\0\x0a\0\0\0\x05\0\0\0\x37\0\0\0\0\0\0\0\x01\0\0\0\x1d\0\0\0\x4c\0\0\0\0\0\0\0\
\x0a\0\0\0\x06\0\0\0\x53\0\0\0\0\0\0\0\x0a\0\0\0\x07\0\0\0\x5a\0\0\0\0\0\0\0\
\x0a\0\0\0\x08\0\0\0\x66\0\0\0\0\0\0\0\x01\0\0\0\x1f\0\0\0\x6f\0\0\0\0\0\0\0\
\x0a\0\0\0\x09\0\0\0\x76\0\0\0\0\0\0\0\x01\0\0\0\x17\0\0\0\x84\0\0\0\0\0\0\0\
\x0a\0\0\0\x0a\0\0\0\x8f\0\0\0\0\0\0\0\x0a\0\0\0\x0b\0\0\0\x9b\0\0\0\0\0\0\0\
\x01\0\0\0\x18\0\0\0\xa4\0\0\0\0\0\0\0\x0a\0\0\0\x0c\0\0\0\xaf\0\0\0\0\0\0\0\
\x0a\0\0\0\x19\0\0\0\xb3\0\0\0\0\0\0\0\x0a\0\0\0\x0d\0\0\0\xd0\0\0\0\0\0\0\0\
\x0a\0\0\0\x0e\0\0\0\xea\0\0\0\0\0\0\0\x0a\0\0\0\x0f\0\0\0\xf1\0\0\0\0\0\0\0\
\x0a\0\0\0\x10\0\0\0\xf8\0\0\0\0\0\0\0\x0a\0\0\0\x11\0\0\0\x19\x01\0\0\0\0\0\0\
\x0a\0\0\0\x12\0\0\0\x29\x01\0\0\0\0\0\0\x0a\0\0\0\x13\0\0\0\x30\x01\0\0\0\0\0\
\0\x0a\0\0\0\x14\0\0\0\xf0\0\0\0\0\0\0\0\0\0\0\0\x1f\0\0\0\x08\x01\0\0\0\0\0\0\
\x0a\0\0\0\x18\0\0\0\x20\x01\0\0\0\0\0\0\0\0\0\0\x1d\0\0\0\x2c\0\0\0\0\0\0\0\0\
\0\0\0\x17\0\0\0\x40\0\0\0\0\0\0\0\0\0\0\0\x17\0\0\0\x50\0\0\0\0\0\0\0\0\0\0\0\
\x17\0\0\0\x60\0\0\0\0\0\0\0\0\0\0\0\x17\0\0\0\x70\0\0\0\0\0\0\0\0\0\0\0\x17\0\
\0\0\x80\0\0\0\0\0\0\0\0\0\0\0\x17\0\0\0\x90\0\0\0\0\0\0\0\0\0\0\0\x17\0\0\0\
\x14\0\0\0\0\0\0\0\x0a\0\0\0\x1b\0\0\0\x18\0\0\0\0\0\0\0\x01\0\0\0\x17\0\0\0\
\x7d\0\0\0\0\0\0\0\x01\0\0\0\x17\0\0\0\x1e\x1d\x16\0\x2e\x64\x65\x62\x75\x67\
\x5f\x61\x62\x62\x72\x65\x76\0\x2e\x74\x65\x78\x74\0\x2e\x72\x65\x6c\x2e\x42\
\x54\x46\x2e\x65\x78\x74\0\x68\x61\x6e\x64\x6c\x65\x5f\x74\x70\x2e\x5f\x5f\x5f\
\x5f\x66\x6d\x74\0\x2e\x62\x73\x73\0\x2e\x64\x65\x62\x75\x67\x5f\x73\x74\x72\0\
\x2e\x72\x65\x6c\x74\x70\x2f\x72\x61\x77\x5f\x73\x79\x73\x63\x61\x6c\x6c\x73\
\x2f\x73\x79\x73\x5f\x65\x6e\x74\x65\x72\0\x68\x61\x6e\x64\x6c\x65\x5f\x74\x70\
\0\x2e\x72\x65\x6c\x2e\x64\x65\x62\x75\x67\x5f\x69\x6e\x66\x6f\0\x2e\x6c\x6c\
\x76\x6d\x5f\x61\x64\x64\x72\x73\x69\x67\0\x6c\x69\x63\x65\x6e\x73\x65\0\x2e\
\x72\x65\x6c\x2e\x64\x65\x62\x75\x67\x5f\x6c\x69\x6e\x65\0\x2e\x72\x65\x6c\x2e\
\x64\x65\x62\x75\x67\x5f\x66\x72\x61\x6d\x65\0\x6d\x79\x5f\x70\x69\x64\0\x2e\
\x64\x65\x62\x75\x67\x5f\x6c\x6f\x63\0\x68\x65\x6c\x6c\x6f\x2e\x62\x70\x66\x2e\
\x63\0\x2e\x73\x74\x72\x74\x61\x62\0\x2e\x73\x79\x6d\x74\x61\x62\0\x2e\x72\x6f\
\x64\x61\x74\x61\0\x2e\x72\x65\x6c\x2e\x42\x54\x46\0\x4c\x49\x43\x45\x4e\x53\
\x45\0\x4c\x42\x42\x30\x5f\x32\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\
\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\
\0\0\0\0\0\0\xd1\0\0\0\x03\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\xc3\x0d\0\0\0\
\0\0\0\x01\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\
\x0f\0\0\0\x01\0\0\0\x06\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x40\0\0\0\0\0\0\0\0\0\0\
\0\0\0\0\0\0\0\0\0\0\0\0\0\x04\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x48\0\0\0\x01\0\0\
\0\x06\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x40\0\0\0\0\0\0\0\x68\0\0\0\0\0\0\0\0\0\0\
\0\0\0\0\0\x08\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x44\0\0\0\x09\0\0\0\0\0\0\0\0\0\0\
\0\0\0\0\0\0\0\0\0\x20\x0b\0\0\0\0\0\0\x20\0\0\0\0\0\0\0\x16\0\0\0\x03\0\0\0\
\x08\0\0\0\0\0\0\0\x10\0\0\0\0\0\0\0\x8a\0\0\0\x01\0\0\0\x03\0\0\0\0\0\0\0\0\0\
\0\0\0\0\0\0\xa8\0\0\0\0\0\0\0\x0d\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x01\0\0\0\0\0\
\0\0\0\0\0\0\0\0\0\0\x34\0\0\0\x08\0\0\0\x03\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\xb8\
\0\0\0\0\0\0\0\x04\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x04\0\0\0\0\0\0\0\0\0\0\0\0\0\
\0\0\xe1\0\0\0\x01\0\0\0\x02\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\xb8\0\0\0\0\0\0\0\
\x1c\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\xba\0\0\0\
\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\xd4\0\0\0\0\0\0\0\x23\0\0\0\0\0\0\0\
\0\0\0\0\0\0\0\0\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x01\0\0\0\x01\0\0\0\0\0\0\0\
\0\0\0\0\0\0\0\0\0\0\0\0\xf7\0\0\0\0\0\0\0\xe8\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\
\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x70\0\0\0\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\
\0\0\0\0\xdf\x01\0\0\0\0\0\0\x38\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x01\0\0\0\0\0\
\0\0\0\0\0\0\0\0\0\0\x6c\0\0\0\x09\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x40\
\x0b\0\0\0\0\0\0\xb0\x01\0\0\0\0\0\0\x16\0\0\0\x0a\0\0\0\x08\0\0\0\0\0\0\0\x10\
\0\0\0\0\0\0\0\x39\0\0\0\x01\0\0\0\x30\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x17\x03\0\
\0\0\0\0\0\x3b\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x01\0\0\0\0\0\0\0\x01\0\0\0\0\0\
\0\0\xed\0\0\0\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x52\x04\0\0\0\0\0\0\
\x59\x02\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\xe9\0\0\
\0\x09\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\xf0\x0c\0\0\0\0\0\0\x30\0\0\0\0\0\
\0\0\x16\0\0\0\x0d\0\0\0\x08\0\0\0\0\0\0\0\x10\0\0\0\0\0\0\0\x19\0\0\0\x01\0\0\
\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\xab\x06\0\0\0\0\0\0\xa0\0\0\0\0\0\0\0\0\0\0\
\0\0\0\0\0\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x15\0\0\0\x09\0\0\0\0\0\0\0\0\0\0\
\0\0\0\0\0\0\0\0\0\x20\x0d\0\0\0\0\0\0\x70\0\0\0\0\0\0\0\x16\0\0\0\x0f\0\0\0\
\x08\0\0\0\0\0\0\0\x10\0\0\0\0\0\0\0\xa6\0\0\0\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\
\0\0\0\0\0\x50\x07\0\0\0\0\0\0\x28\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x08\0\0\0\0\0\
\0\0\0\0\0\0\0\0\0\0\xa2\0\0\0\x09\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x90\
\x0d\0\0\0\0\0\0\x20\0\0\0\0\0\0\0\x16\0\0\0\x11\0\0\0\x08\0\0\0\0\0\0\0\x10\0\
\0\0\0\0\0\0\x96\0\0\0\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x78\x07\0\0\0\
\0\0\0\xa4\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x92\
\0\0\0\x09\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\xb0\x0d\0\0\0\0\0\0\x10\0\0\0\
\0\0\0\0\x16\0\0\0\x13\0\0\0\x08\0\0\0\0\0\0\0\x10\0\0\0\0\0\0\0\x7c\0\0\0\x03\
\x4c\xff\x6f\0\0\0\x80\0\0\0\0\0\0\0\0\0\0\0\0\xc0\x0d\0\0\0\0\0\0\x03\0\0\0\0\
\0\0\0\x16\0\0\0\0\0\0\0\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\xd9\0\0\0\x02\0\0\0\
\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x20\x08\0\0\0\0\0\0\0\x03\0\0\0\0\0\0\x01\0\0\
\0\x1d\0\0\0\x08\0\0\0\0\0\0\0\x18\0\0\0\0\0\0\0";

	obj->skeleton = s;
	return 0;
err:
	bpf_object__destroy_skeleton(s);
	return -ENOMEM;
}

#endif /* __HELLO_BPF_SKEL_H__ */