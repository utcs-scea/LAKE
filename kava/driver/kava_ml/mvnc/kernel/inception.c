#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <asm/segment.h>
#include <asm/uaccess.h>
#include <linux/buffer_head.h>
#include <asm/fpu/api.h>

#include "mvnc_nw.h"
#include "shared_memory.h"

#define description_string "Kernel implementation of mvnc."
#define maintainer_string "Bodun Hu"

MODULE_AUTHOR(maintainer_string);
MODULE_DESCRIPTION(description_string);
MODULE_VERSION("0.01");
MODULE_LICENSE("GPL");  /* required for kernel_fpu_begin */


#define BUF_LEN 1024
static char input_graph[BUF_LEN] __initdata;
module_param_string(input_graph, input_graph, BUF_LEN, S_IRUGO);

static char input_labels[BUF_LEN] __initdata;
module_param_string(input_labels, input_labels, BUF_LEN, S_IRUGO);

static char input_image[BUF_LEN] __initdata;
module_param_string(input_image, input_image, BUF_LEN, S_IRUGO);

static int total_images;
module_param(total_images, int , 0);

static int batch_mode;
module_param(batch_mode, int , 0);

static int fifo_length;
module_param(fifo_length, int, 0);

#define BUFFER_SIZE 60

#define MEASURE_END2END_TIME
#define MEASURE_INFERENCE_TIME
#define MEASURE_GRAPH_INIT_TIME
#define MEASURE_H2D_TIME
#define MEASURE_D2H_TIME
#define MEASURE_INFERENCE_CALL

#define njAllocMem vmalloc
#define njFreeMem vfree
#define njCopyMem  memcpy
#define njFillMem memset


#ifndef NJ_CHROMA_FILTER
    #define NJ_CHROMA_FILTER 1
#endif

#if NJ_CHROMA_FILTER

#define CF4A (-9)
#define CF4B (111)
#define CF4C (29)
#define CF4D (-3)
#define CF3A (28)
#define CF3B (109)
#define CF3C (-9)
#define CF3X (104)
#define CF3Y (27)
#define CF3Z (-3)
#define CF2A (139)
#define CF2B (-11)
#define CF(x) njClip(((x) + 64) >> 7)

#endif


#define W1 2841
#define W2 2676
#define W3 2408
#define W5 1609
#define W6 1108
#define W7 565

#define njThrow(e) do { nj.error = e; return; } while (0)
#define njCheckError() do { if (nj.error) return; } while (0)

typedef enum _nj_result {
    NJ_OK = 0,        // no error, decoding successful
    NJ_NO_JPEG,       // not a JPEG file
    NJ_UNSUPPORTED,   // unsupported format
    NJ_OUT_OF_MEM,    // out of memory
    NJ_INTERNAL_ERR,  // internal error
    NJ_SYNTAX_ERROR,  // syntax error
    __NJ_FINISHED,    // used internally, will never be reported
} nj_result_t;

typedef struct _nj_code {
    unsigned char bits, code;
} nj_vlc_code_t;

typedef struct _nj_cmp {
    int cid;
    int ssx, ssy;
    int width, height;
    int stride;
    int qtsel;
    int actabsel, dctabsel;
    int dcpred;
    unsigned char *pixels;
} nj_component_t;

typedef struct _nj_ctx {
    nj_result_t error;
    const unsigned char *pos;
    int size;
    int length;
    int width, height;
    int mbwidth, mbheight;
    int mbsizex, mbsizey;
    int ncomp;
    nj_component_t comp[3];
    int qtused, qtavail;
    unsigned char qtab[4][64];
    nj_vlc_code_t vlctab[4][65536];
    int buf, bufbits;
    int block[64];
    int rstinterval;
    unsigned char *rgb;
} nj_context_t;

static nj_context_t nj;

static const char njZZ[64] = { 0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18,
11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35,
42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45,
38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 };

static inline unsigned char njClip(const int x) {
    return (x < 0) ? 0 : ((x > 0xFF) ? 0xFF : (unsigned char) x);
}

unsigned char* njGetImage(void) { return (nj.ncomp == 1) ? nj.comp[0].pixels : nj.rgb; }
int njGetImageSize(void)        { return nj.width * nj.height * nj.ncomp; }

unsigned long file_length(struct file* file, const char *path) {
    mm_segment_t old_fs;
    unsigned long input_size;
    struct kstat *stat;

	old_fs = get_fs();
	set_fs(get_ds());
  
    stat = (struct kstat *)vmalloc(sizeof(struct kstat));
    if (!stat) return 0;
    vfs_llseek(file, 0, SEEK_END); 
    vfs_stat(path, stat);
    input_size = (unsigned long)stat->size;
    vfs_llseek(file, 0, SEEK_SET); 
    vfree(stat);
    set_fs(old_fs);
    
    return input_size;
}

char *load_file(struct file *file, unsigned long length) {
    char *buffer;
    loff_t pos = 0;
    ssize_t r;
    buffer = (char *)kava_alloc((size_t)(length + 1));
    BUG_ON(buffer == NULL);

    //loff_t pos = vfs_llseek(file, 0, SEEK_SET); 
    //ssize_t r = kernel_read(file, (void *)buffer, (size_t)length, &pos);
    r = kernel_read(file, buffer, (size_t)length, &pos);
    //vfs_llseek(file, 0, SEEK_SET); 
    //if (r < 0) return NULL;
    return buffer;
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

static inline void njRowIDCT(int* blk) {
    int x0, x1, x2, x3, x4, x5, x6, x7, x8;
    if (!((x1 = blk[4] << 11)
        | (x2 = blk[6])
        | (x3 = blk[2])
        | (x4 = blk[1])
        | (x5 = blk[7])
        | (x6 = blk[5])
        | (x7 = blk[3])))
    {
        blk[0] = blk[1] = blk[2] = blk[3] = blk[4] = blk[5] = blk[6] = blk[7] = blk[0] << 3;
        return;
    }
    x0 = (blk[0] << 11) + 128;
    x8 = W7 * (x4 + x5);
    x4 = x8 + (W1 - W7) * x4;
    x5 = x8 - (W1 + W7) * x5;
    x8 = W3 * (x6 + x7);
    x6 = x8 - (W3 - W5) * x6;
    x7 = x8 - (W3 + W5) * x7;
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6 * (x3 + x2);
    x2 = x1 - (W2 + W6) * x2;
    x3 = x1 + (W2 - W6) * x3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;
    blk[0] = (x7 + x1) >> 8;
    blk[1] = (x3 + x2) >> 8;
    blk[2] = (x0 + x4) >> 8;
    blk[3] = (x8 + x6) >> 8;
    blk[4] = (x8 - x6) >> 8;
    blk[5] = (x0 - x4) >> 8;
    blk[6] = (x3 - x2) >> 8;
    blk[7] = (x7 - x1) >> 8;
}

static inline void njColIDCT(const int* blk, unsigned char *out, int stride) {
    int x0, x1, x2, x3, x4, x5, x6, x7, x8;
    if (!((x1 = blk[8*4] << 8)
        | (x2 = blk[8*6])
        | (x3 = blk[8*2])
        | (x4 = blk[8*1])
        | (x5 = blk[8*7])
        | (x6 = blk[8*5])
        | (x7 = blk[8*3])))
    {
        x1 = njClip(((blk[0] + 32) >> 6) + 128);
        for (x0 = 8;  x0;  --x0) {
            *out = (unsigned char) x1;
            out += stride;
        }
        return;
    }
    x0 = (blk[0] << 8) + 8192;
    x8 = W7 * (x4 + x5) + 4;
    x4 = (x8 + (W1 - W7) * x4) >> 3;
    x5 = (x8 - (W1 + W7) * x5) >> 3;
    x8 = W3 * (x6 + x7) + 4;
    x6 = (x8 - (W3 - W5) * x6) >> 3;
    x7 = (x8 - (W3 + W5) * x7) >> 3;
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6 * (x3 + x2) + 4;
    x2 = (x1 - (W2 + W6) * x2) >> 3;
    x3 = (x1 + (W2 - W6) * x3) >> 3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;
    *out = njClip(((x7 + x1) >> 14) + 128);  out += stride;
    *out = njClip(((x3 + x2) >> 14) + 128);  out += stride;
    *out = njClip(((x0 + x4) >> 14) + 128);  out += stride;
    *out = njClip(((x8 + x6) >> 14) + 128);  out += stride;
    *out = njClip(((x8 - x6) >> 14) + 128);  out += stride;
    *out = njClip(((x0 - x4) >> 14) + 128);  out += stride;
    *out = njClip(((x3 - x2) >> 14) + 128);  out += stride;
    *out = njClip(((x7 - x1) >> 14) + 128);
}

static int njShowBits(int bits) {
    unsigned char newbyte;
    unsigned char marker;
    if (!bits) return 0;
    while (nj.bufbits < bits) {
        if (nj.size <= 0) {
            nj.buf = (nj.buf << 8) | 0xFF;
            nj.bufbits += 8;
            continue;
        }
        newbyte = *nj.pos++;
        nj.size--;
        nj.bufbits += 8;
        nj.buf = (nj.buf << 8) | newbyte;
        if (newbyte == 0xFF) {
            if (nj.size) {
                marker = *nj.pos++;
                nj.size--;
                switch (marker) {
                    case 0x00:
                    case 0xFF:
                        break;
                    case 0xD9: nj.size = 0; break;
                    default:
                        if ((marker & 0xF8) != 0xD0)
                            nj.error = NJ_SYNTAX_ERROR;
                        else {
                            nj.buf = (nj.buf << 8) | marker;
                            nj.bufbits += 8;
                        }
                }
            } else
                nj.error = NJ_SYNTAX_ERROR;
        }
    }
    return (nj.buf >> (nj.bufbits - bits)) & ((1 << bits) - 1);
}

static inline void njSkipBits(int bits) {
    if (nj.bufbits < bits)
        (void) njShowBits(bits);
    nj.bufbits -= bits;
}

static inline int njGetBits(int bits) {
    int res = njShowBits(bits);
    njSkipBits(bits);
    return res;
}

static inline void njByteAlign(void) {
    nj.bufbits &= 0xF8;
}

static void njSkip(int count) {
    nj.pos += count;
    nj.size -= count;
    nj.length -= count;
    if (nj.size < 0) nj.error = NJ_SYNTAX_ERROR;
}

static inline unsigned short njDecode16(const unsigned char *pos) {
    return (pos[0] << 8) | pos[1];
}


static void njDecodeLength(void) {
    if (nj.size < 2) {
        printk(KERN_ALERT "njDecodeLength: nj.size less than 2!\n");
        njThrow(NJ_SYNTAX_ERROR);
    }
    nj.length = njDecode16(nj.pos);
    if (nj.length > nj.size) {
        printk(KERN_ALERT "njDecode16 failed!\n");
        njThrow(NJ_SYNTAX_ERROR);
    }
    njSkip(2);
}

static inline void njSkipMarker(void) {
    njDecodeLength();
    njSkip(nj.length);
}

void njInit(void) {
    memset(&nj, 0, sizeof(nj_context_t));
}

void njDone(void) {
    int i;
    for (i = 0;  i < 3;  ++i)
        if (nj.comp[i].pixels) njFreeMem((void*) nj.comp[i].pixels);
    if (nj.rgb) njFreeMem((void*) nj.rgb);
    njInit();
}


static inline void njUpsampleH(nj_component_t* c) {
    const int xmax = c->width - 3;
    unsigned char *out, *lin, *lout;
    int x, y;
    out = (unsigned char*) njAllocMem((c->width * c->height) << 1);
    if (!out) njThrow(NJ_OUT_OF_MEM);
    lin = c->pixels;
    lout = out;
    for (y = c->height;  y;  --y) {
        lout[0] = CF(CF2A * lin[0] + CF2B * lin[1]);
        lout[1] = CF(CF3X * lin[0] + CF3Y * lin[1] + CF3Z * lin[2]);
        lout[2] = CF(CF3A * lin[0] + CF3B * lin[1] + CF3C * lin[2]);
        for (x = 0;  x < xmax;  ++x) {
            lout[(x << 1) + 3] = CF(CF4A * lin[x] + CF4B * lin[x + 1] + CF4C * lin[x + 2] + CF4D * lin[x + 3]);
            lout[(x << 1) + 4] = CF(CF4D * lin[x] + CF4C * lin[x + 1] + CF4B * lin[x + 2] + CF4A * lin[x + 3]);
        }
        lin += c->stride;
        lout += c->width << 1;
        lout[-3] = CF(CF3A * lin[-1] + CF3B * lin[-2] + CF3C * lin[-3]);
        lout[-2] = CF(CF3X * lin[-1] + CF3Y * lin[-2] + CF3Z * lin[-3]);
        lout[-1] = CF(CF2A * lin[-1] + CF2B * lin[-2]);
    }
    c->width <<= 1;
    c->stride = c->width;
    njFreeMem((void*)c->pixels);
    c->pixels = out;
}

static inline void njUpsampleV(nj_component_t* c) {
    const int w = c->width, s1 = c->stride, s2 = s1 + s1;
    unsigned char *out, *cin, *cout;
    int x, y;
    out = (unsigned char*) njAllocMem((c->width * c->height) << 1);
    if (!out) njThrow(NJ_OUT_OF_MEM);
    for (x = 0;  x < w;  ++x) {
        cin = &c->pixels[x];
        cout = &out[x];
        *cout = CF(CF2A * cin[0] + CF2B * cin[s1]);  cout += w;
        *cout = CF(CF3X * cin[0] + CF3Y * cin[s1] + CF3Z * cin[s2]);  cout += w;
        *cout = CF(CF3A * cin[0] + CF3B * cin[s1] + CF3C * cin[s2]);  cout += w;
        cin += s1;
        for (y = c->height - 3;  y;  --y) {
            *cout = CF(CF4A * cin[-s1] + CF4B * cin[0] + CF4C * cin[s1] + CF4D * cin[s2]);  cout += w;
            *cout = CF(CF4D * cin[-s1] + CF4C * cin[0] + CF4B * cin[s1] + CF4A * cin[s2]);  cout += w;
            cin += s1;
        }
        cin += s1;
        *cout = CF(CF3A * cin[0] + CF3B * cin[-s1] + CF3C * cin[-s2]);  cout += w;
        *cout = CF(CF3X * cin[0] + CF3Y * cin[-s1] + CF3Z * cin[-s2]);  cout += w;
        *cout = CF(CF2A * cin[0] + CF2B * cin[-s1]);
    }
    c->height <<= 1;
    c->stride = c->width;
    njFreeMem((void*) c->pixels);
    c->pixels = out;
}


static inline void njConvert(void) {
    int i;
    nj_component_t* c;
    for (i = 0, c = nj.comp;  i < nj.ncomp;  ++i, ++c) {
        #if NJ_CHROMA_FILTER
            while ((c->width < nj.width) || (c->height < nj.height)) {
                if (c->width < nj.width) njUpsampleH(c);
                njCheckError();
                if (c->height < nj.height) njUpsampleV(c);
                njCheckError();
            }
        #else
            if ((c->width < nj.width) || (c->height < nj.height))
                njUpsample(c);
        #endif
        if ((c->width < nj.width) || (c->height < nj.height)) njThrow(NJ_INTERNAL_ERR);
    }
    if (nj.ncomp == 3) {
        // convert to RGB
        int x, yy;
        unsigned char *prgb = nj.rgb;
        const unsigned char *py  = nj.comp[0].pixels;
        const unsigned char *pcb = nj.comp[1].pixels;
        const unsigned char *pcr = nj.comp[2].pixels;
        for (yy = nj.height;  yy;  --yy) {
            for (x = 0;  x < nj.width;  ++x) {
                register int y = py[x] << 8;
                register int cb = pcb[x] - 128;
                register int cr = pcr[x] - 128;
                *prgb++ = njClip((y            + 359 * cr + 128) >> 8);
                *prgb++ = njClip((y -  88 * cb - 183 * cr + 128) >> 8);
                *prgb++ = njClip((y + 454 * cb            + 128) >> 8);
            }
            py += nj.comp[0].stride;
            pcb += nj.comp[1].stride;
            pcr += nj.comp[2].stride;
        }
    } else if (nj.comp[0].width != nj.comp[0].stride) {
        // grayscale -> only remove stride
        unsigned char *pin = &nj.comp[0].pixels[nj.comp[0].stride];
        unsigned char *pout = &nj.comp[0].pixels[nj.comp[0].width];
        int y;
        for (y = nj.comp[0].height - 1;  y;  --y) {
            njCopyMem(pout, pin, nj.comp[0].width);
            pin += nj.comp[0].stride;
            pout += nj.comp[0].width;
        }
        nj.comp[0].stride = nj.comp[0].width;
    }
}

static inline void njDecodeSOF(void) {
    int i, ssxmax = 0, ssymax = 0;
    nj_component_t* c;
    njDecodeLength();
    njCheckError();
    if (nj.length < 9) njThrow(NJ_SYNTAX_ERROR);
    if (nj.pos[0] != 8) njThrow(NJ_UNSUPPORTED);
    nj.height = njDecode16(nj.pos+1);
    nj.width = njDecode16(nj.pos+3);
    if (!nj.width || !nj.height) njThrow(NJ_SYNTAX_ERROR);
    nj.ncomp = nj.pos[5];
    njSkip(6);
    switch (nj.ncomp) {
        case 1:
        case 3:
            break;
        default:
            njThrow(NJ_UNSUPPORTED);
    }
    if (nj.length < (nj.ncomp * 3)) njThrow(NJ_SYNTAX_ERROR);
    for (i = 0, c = nj.comp;  i < nj.ncomp;  ++i, ++c) {
        c->cid = nj.pos[0];
        if (!(c->ssx = nj.pos[1] >> 4)) njThrow(NJ_SYNTAX_ERROR);
        if (c->ssx & (c->ssx - 1)) njThrow(NJ_UNSUPPORTED);  // non-power of two
        if (!(c->ssy = nj.pos[1] & 15)) njThrow(NJ_SYNTAX_ERROR);
        if (c->ssy & (c->ssy - 1)) njThrow(NJ_UNSUPPORTED);  // non-power of two
        if ((c->qtsel = nj.pos[2]) & 0xFC) njThrow(NJ_SYNTAX_ERROR);
        njSkip(3);
        nj.qtused |= 1 << c->qtsel;
        if (c->ssx > ssxmax) ssxmax = c->ssx;
        if (c->ssy > ssymax) ssymax = c->ssy;
    }
    if (nj.ncomp == 1) {
        c = nj.comp;
        c->ssx = c->ssy = ssxmax = ssymax = 1;
    }
    nj.mbsizex = ssxmax << 3;
    nj.mbsizey = ssymax << 3;
    nj.mbwidth = (nj.width + nj.mbsizex - 1) / nj.mbsizex;
    nj.mbheight = (nj.height + nj.mbsizey - 1) / nj.mbsizey;
    for (i = 0, c = nj.comp;  i < nj.ncomp;  ++i, ++c) {
        c->width = (nj.width * c->ssx + ssxmax - 1) / ssxmax;
        c->height = (nj.height * c->ssy + ssymax - 1) / ssymax;
        c->stride = nj.mbwidth * c->ssx << 3;
        if (((c->width < 3) && (c->ssx != ssxmax)) || ((c->height < 3) && (c->ssy != ssymax))) njThrow(NJ_UNSUPPORTED);
        if (!(c->pixels = (unsigned char*) njAllocMem(c->stride * nj.mbheight * c->ssy << 3))) njThrow(NJ_OUT_OF_MEM);
    }
    if (nj.ncomp == 3) {
        nj.rgb = (unsigned char*) njAllocMem(nj.width * nj.height * nj.ncomp);
        if (!nj.rgb) njThrow(NJ_OUT_OF_MEM);
    }
    njSkip(nj.length);
}

static inline void njDecodeDHT(void) {
    int codelen, currcnt, remain, spread, i, j;
    nj_vlc_code_t *vlc;
    static unsigned char counts[16];
    njDecodeLength();
    njCheckError();
    while (nj.length >= 17) {
        i = nj.pos[0];
        if (i & 0xEC) njThrow(NJ_SYNTAX_ERROR);
        if (i & 0x02) njThrow(NJ_UNSUPPORTED);
        i = (i | (i >> 3)) & 3;  // combined DC/AC + tableid value
        for (codelen = 1;  codelen <= 16;  ++codelen)
            counts[codelen - 1] = nj.pos[codelen];
        njSkip(17);
        vlc = &nj.vlctab[i][0];
        remain = spread = 65536;
        for (codelen = 1;  codelen <= 16;  ++codelen) {
            spread >>= 1;
            currcnt = counts[codelen - 1];
            if (!currcnt) continue;
            if (nj.length < currcnt) njThrow(NJ_SYNTAX_ERROR);
            remain -= currcnt << (16 - codelen);
            if (remain < 0) njThrow(NJ_SYNTAX_ERROR);
            for (i = 0;  i < currcnt;  ++i) {
                register unsigned char code = nj.pos[i];
                for (j = spread;  j;  --j) {
                    vlc->bits = (unsigned char) codelen;
                    vlc->code = code;
                    ++vlc;
                }
            }
            njSkip(currcnt);
        }
        while (remain--) {
            vlc->bits = 0;
            ++vlc;
        }
    }
    if (nj.length) njThrow(NJ_SYNTAX_ERROR);
}

static int njGetVLC(nj_vlc_code_t* vlc, unsigned char* code) {
    int value = njShowBits(16);
    int bits = vlc[value].bits;
    if (!bits) { nj.error = NJ_SYNTAX_ERROR; return 0; }
    njSkipBits(bits);
    value = vlc[value].code;
    if (code) *code = (unsigned char) value;
    bits = value & 15;
    if (!bits) return 0;
    value = njGetBits(bits);
    if (value < (1 << (bits - 1)))
        value += ((-1) << bits) + 1;
    return value;
}

static inline void njDecodeBlock(nj_component_t* c, unsigned char* out) {
    unsigned char code = 0;
    int value, coef = 0;
    njFillMem(nj.block, 0, sizeof(nj.block));
    c->dcpred += njGetVLC(&nj.vlctab[c->dctabsel][0], NULL);
    nj.block[0] = (c->dcpred) * nj.qtab[c->qtsel][0];
    do {
        value = njGetVLC(&nj.vlctab[c->actabsel][0], &code);
        if (!code) break;  // EOB
        if (!(code & 0x0F) && (code != 0xF0)) njThrow(NJ_SYNTAX_ERROR);
        coef += (code >> 4) + 1;
        if (coef > 63) njThrow(NJ_SYNTAX_ERROR);
        nj.block[(int) njZZ[coef]] = value * nj.qtab[c->qtsel][coef];
    } while (coef < 63);
    for (coef = 0;  coef < 64;  coef += 8)
        njRowIDCT(&nj.block[coef]);
    for (coef = 0;  coef < 8;  ++coef)
        njColIDCT(&nj.block[coef], &out[coef], c->stride);
}


static inline void njDecodeScan(void) {
    int i, mbx, mby, sbx, sby;
    int rstcount = nj.rstinterval, nextrst = 0;
    nj_component_t* c;
    njDecodeLength();
    njCheckError();
    if (nj.length < (4 + 2 * nj.ncomp)) njThrow(NJ_SYNTAX_ERROR);
    if (nj.pos[0] != nj.ncomp) njThrow(NJ_UNSUPPORTED);
    njSkip(1);
    for (i = 0, c = nj.comp;  i < nj.ncomp;  ++i, ++c) {
        if (nj.pos[0] != c->cid) njThrow(NJ_SYNTAX_ERROR);
        if (nj.pos[1] & 0xEE) njThrow(NJ_SYNTAX_ERROR);
        c->dctabsel = nj.pos[1] >> 4;
        c->actabsel = (nj.pos[1] & 1) | 2;
        njSkip(2);
    }
    if (nj.pos[0] || (nj.pos[1] != 63) || nj.pos[2]) njThrow(NJ_UNSUPPORTED);
    njSkip(nj.length);
    for (mbx = mby = 0;;) {
        for (i = 0, c = nj.comp;  i < nj.ncomp;  ++i, ++c)
            for (sby = 0;  sby < c->ssy;  ++sby)
                for (sbx = 0;  sbx < c->ssx;  ++sbx) {
                    njDecodeBlock(c, &c->pixels[((mby * c->ssy + sby) * c->stride + mbx * c->ssx + sbx) << 3]);
                    njCheckError();
                }
        if (++mbx >= nj.mbwidth) {
            mbx = 0;
            if (++mby >= nj.mbheight) break;
        }
        if (nj.rstinterval && !(--rstcount)) {
            njByteAlign();
            i = njGetBits(16);
            if (((i & 0xFFF8) != 0xFFD0) || ((i & 7) != nextrst)) njThrow(NJ_SYNTAX_ERROR);
            nextrst = (nextrst + 1) & 7;
            rstcount = nj.rstinterval;
            for (i = 0;  i < 3;  ++i)
                nj.comp[i].dcpred = 0;
        }
    }
    nj.error = __NJ_FINISHED;
}


static inline void njDecodeDQT(void) {
    int i;
    unsigned char *t;
    njDecodeLength();
    njCheckError();
    while (nj.length >= 65) {
        i = nj.pos[0];
        if (i & 0xFC) njThrow(NJ_SYNTAX_ERROR);
        nj.qtavail |= 1 << i;
        t = &nj.qtab[i][0];
        for (i = 0;  i < 64;  ++i)
            t[i] = nj.pos[i + 1];
        njSkip(65);
    }
    if (nj.length) njThrow(NJ_SYNTAX_ERROR);
}

static inline void njDecodeDRI(void) {
    njDecodeLength();
    njCheckError();
    if (nj.length < 2) njThrow(NJ_SYNTAX_ERROR);
    nj.rstinterval = njDecode16(nj.pos);
    njSkip(nj.length);
}

nj_result_t njDecode(const void* jpeg, const int size) {
    njDone();
    nj.pos = (const unsigned char*) jpeg;
    nj.size = size & 0x7FFFFFFF;
    if (nj.size < 2) {
        printk(KERN_ALERT "njDecode: nj.size less than 2!\n");
        return NJ_NO_JPEG;
    }
    if ((nj.pos[0] ^ 0xFF) | (nj.pos[1] ^ 0xD8)) {
        printk(KERN_ALERT "njDecode: NO_JPEG err!\n");
        return NJ_NO_JPEG;
    }
    njSkip(2);
    while (!nj.error) {
        printk(KERN_ALERT "it\n");
        if ((nj.size < 2) || (nj.pos[0] != 0xFF)) {
            printk(KERN_ALERT "njDecode: MJ syntax err\n");
            return NJ_SYNTAX_ERROR;
        }
        njSkip(2);
        switch (nj.pos[-1]) {
            case 0xC0: printk(KERN_ALERT "SOF\n"); njDecodeSOF(); break;
            case 0xC4: printk(KERN_ALERT "DHT\n"); njDecodeDHT(); break;
            case 0xDB: printk(KERN_ALERT "DQT\n"); njDecodeDQT(); break;
            case 0xDD: printk(KERN_ALERT "DRI\n"); njDecodeDRI(); break;
            case 0xDA: printk(KERN_ALERT "Scan\n"); njDecodeScan(); break;
            case 0xFE: printk(KERN_ALERT "SskipMarker\n"); njSkipMarker(); break;
            default: {
                if ((nj.pos[-1] & 0xF0) == 0xE0) {
                    njSkipMarker();
                }
                else {
                    printk(KERN_ALERT "njDecode: nj unsupported!\n");
                    return NJ_UNSUPPORTED; 
                }
            }
        }
    }

    if (nj.error != __NJ_FINISHED) {
        printk(KERN_ALERT "njDecode: nj not finished!\n");
        return nj.error;
    }
    nj.error = NJ_OK;
    njConvert();
    return nj.error;
}

static int mvnc_single(void) {
#ifdef MEASURE_END2END_TIME
    // End to end measurement
    struct timespec micro_end2end_start, micro_end2end_stop;
    long total_end2end_micro = 0;
#endif

#ifdef MEASURE_INFERENCE_TIME
    // inference measurement
    struct timespec micro_inference_start, micro_inference_stop;
    long total_inference_micro = 0;
#endif

#ifdef MEASURE_INFERENCE_CALL
    struct timespec micro_inference_call_start, micro_inference_call_stop;
    long total_inference_call_micro = 0;
#endif

#ifdef MEASURE_GRAPH_INIT_TIME
    // graph init measurement
    struct timespec micro_graph_init_start, micro_graph_init_stop;
    long total_graph_init_micro = 0;
#endif

#ifdef MEASURE_H2D_TIME
    // graph init measurement
    struct timespec micro_h2d_start, micro_h2d_stop;
    long total_h2d_micro = 0;
#endif

#ifdef MEASURE_D2H_TIME
    struct timespec micro_d2h_start, micro_d2h_stop;
    long total_d2h_micro = 0;
#endif


    ncStatus_t retCode;
    struct ncDeviceHandle_t *deviceHandle = NULL;
    int attempCounter = 10000;
    struct file *graphFile = NULL;
    struct file *image_file = NULL;
    struct ncGraphHandle_t *graphHandle;
    struct ncFifoHandle_t* inFifoHandle;
    struct ncFifoHandle_t* outFifoHandle;

    unsigned long graphFileLength;
    char *graphFileBuffer = NULL;
    unsigned long image_file_length;
    char *image_file_buffer = NULL;
    nj_result_t result;
    unsigned char *old_image_data = NULL;
    int old_image_data_length;
    unsigned int imageBufferLength;
    float* imageBuffer;
    int i;
    int passes;
    unsigned int outFifoElemSize;
    unsigned int optionSize;
    float* resultData = NULL;

#ifdef MEASURE_END2END_TIME
    getnstimeofday(&micro_end2end_start); 
#endif
    retCode = ncDeviceCreate(0, &deviceHandle);
    while (retCode != NC_OK) {
        if (attempCounter == 0) break;
        retCode = ncDeviceCreate(0, &deviceHandle);
        attempCounter--;
    }

    if (retCode != NC_OK) {
        printk(KERN_INFO "nvDeviceCreate failed.\n");
        return -1;
    }

    retCode = ncDeviceOpen(deviceHandle);
    attempCounter = 10000;
    while (retCode != NC_OK) {
        if (attempCounter == 0) break;
        retCode = ncDeviceOpen(deviceHandle);
        attempCounter--;
    }

    if (retCode != NC_OK) {
        printk(KERN_INFO "ncDeviceOpen failed.\n");
        return -1;
    }

    if (NULL == deviceHandle) {
        printk(KERN_INFO "ncDeviceOpen returned NULL handle\n");
        return -1;
    }

    graphFile = file_open(input_graph, O_RDONLY, 0);
    if (!graphFile) {
        pr_err("Can't open input_graph file.\n");
        return -1;
    }

    // Check file length
    graphFileLength = file_length(graphFile, input_graph);
    if (graphFileLength == 0) {
        printk(KERN_ALERT "graphFileLength failed\n");
        return -1;
    }

    graphFileBuffer = load_file(graphFile, graphFileLength);
    if (graphFileBuffer == NULL) {
        printk(KERN_ALERT "graphFileBuffer failed\n");
        return -1;
    }

#ifdef MEASURE_GRAPH_INIT_TIME
    getnstimeofday(&micro_graph_init_start); 
#endif

    retCode = ncGraphCreate("Inceptionv3-reshape", &graphHandle);
    if (retCode != NC_OK) {
        printk(KERN_ALERT "ncGraphCreate failed.\n");
        return -1;
    }

    retCode = ncGraphAllocateWithFifosEx(deviceHandle, graphHandle, graphFileBuffer, graphFileLength,
                                        &inFifoHandle, NC_FIFO_HOST_WO, total_images, NC_FIFO_FP32,
                                        &outFifoHandle, NC_FIFO_HOST_RO, total_images, NC_FIFO_FP32);
    if (retCode != NC_OK) {
        printk(KERN_ALERT "ncGraphAllocateWithFifosEx failed.\n");
        return -1;
    }

#ifdef MEASURE_GRAPH_INIT_TIME
    getnstimeofday(&micro_graph_init_stop);
    total_graph_init_micro += ELAPSED_TIME_MICRO_SEC(micro_graph_init_start, micro_graph_init_stop);
#endif

    // read tensor from image
    njInit();
    image_file = file_open(input_image, O_RDONLY, 0);
    if (!image_file) {
        pr_err("Can't open input_image file.\n");
        return -1;
    }
    image_file_length = file_length(image_file, input_image);
    if (image_file_length == 0) {
        printk(KERN_ALERT "graphFileLength failed\n");
        return -1;
    }
    image_file_buffer = load_file(image_file, image_file_length);
    if (image_file_buffer == NULL) {
        printk(KERN_ALERT "load image file failed\n");
        return -1;
    }

    result = njDecode(image_file_buffer, image_file_length);
    if (result != NJ_OK) {
        printk(KERN_ALERT "njDecode failed\n");
        return -1;
    } else {
        printk(KERN_ALERT "njDecode success.\n");
    }

    old_image_data = njGetImage();
    old_image_data_length = njGetImageSize();
   
    // convert uint8 to float
    imageBufferLength = old_image_data_length * sizeof(float);
    // printk(KERN_ALERT "nj.width is %d\n", nj.width);
    // printk(KERN_ALERT "nj.height is %d\n", nj.height);
    // printk(KERN_ALERT "nj.ncomp is %d\n", nj.ncomp);
    // printk(KERN_ALERT "imagebufferlength is %d\n", imageBufferLength);
    imageBuffer = kava_alloc(imageBufferLength);

    for (i = 0; i < old_image_data_length; ++i) {
        kernel_fpu_begin();
        imageBuffer[i] = (float)(old_image_data[i])/255.0;
        kernel_fpu_end();
    }

#ifdef MEASURE_INFERENCE_TIME
        getnstimeofday(&micro_inference_start);
#endif

#ifdef MEASURE_H2D_TIME
        getnstimeofday(&micro_h2d_start);
#endif
        // 4. Run the image through the model
        retCode = ncFifoWriteElem(inFifoHandle, imageBuffer, &imageBufferLength, NULL);
        if (retCode != NC_OK) {
            printk(KERN_ALERT "ncFifoWriteElem failed.\n");
            return -1;
        }

#ifdef MEASURE_H2D_TIME
        getnstimeofday(&micro_h2d_stop);
        total_h2d_micro += ELAPSED_TIME_MICRO_SEC(micro_h2d_start,
                micro_h2d_stop);
#endif

#ifdef MEASURE_INFERENCE_CALL
        getnstimeofday(&micro_inference_call_start);
#endif  

        retCode = ncGraphQueueInference(graphHandle, &inFifoHandle, 1,
                                &outFifoHandle, 1);
        if (retCode != NC_OK) {
            printk(KERN_ALERT "ncGraphQueueInference failed.\n");
            return -1;
        }

#ifdef MEASURE_INFERENCE_CALL
        getnstimeofday(&micro_inference_call_stop);
        total_inference_call_micro += ELAPSED_TIME_MICRO_SEC(micro_inference_call_start,
                micro_inference_call_stop);
#endif
        outFifoElemSize = 0;
        optionSize = sizeof(outFifoElemSize);
        retCode = ncFifoGetOption(outFifoHandle, NC_RO_FIFO_ELEMENT_DATA_SIZE,
                &outFifoElemSize, &optionSize);
        if (retCode != NC_OK) {
            printk(KERN_ALERT "ncFifoGetOption failed.\n");
            return -1;
        }

        // Get the output tensor
        resultData = (float*) kava_alloc(outFifoElemSize);

#ifdef MEASURE_D2H_TIME
        getnstimeofday(&micro_d2h_start); 
#endif
        // We don't support userParam...
        retCode = ncFifoReadElem(outFifoHandle, (void*)resultData, &outFifoElemSize, NULL);
        if (retCode != NC_OK) {
            printk(KERN_ALERT "ncFifoReadElem failed.\n");
            return -1;
        }

#ifdef MEASURE_D2H_TIME
        getnstimeofday(&micro_d2h_stop);
        total_d2h_micro += ELAPSED_TIME_MICRO_SEC(micro_d2h_start, micro_d2h_stop);
#endif

        // This may be commented out
        kava_free(resultData);
        file_close(image_file);

#ifdef MEASURE_INFERENCE_TIME
    getnstimeofday(&micro_inference_stop); 
    total_inference_micro += ELAPSED_TIME_MICRO_SEC(micro_inference_start, micro_inference_stop);
#endif

    // Clean up the FIFOs
    ncFifoDestroy(&inFifoHandle);
    ncFifoDestroy(&outFifoHandle);

    // Clean up the graph
    ncGraphDestroy(&graphHandle);

    // Close and clean up the device
    ncDeviceClose(deviceHandle);
    ncDeviceDestroy(&deviceHandle);

#ifdef MEASURE_END2END_TIME
    getnstimeofday(&micro_end2end_stop);
    total_end2end_micro += ELAPSED_TIME_MICRO_SEC(micro_end2end_start, micro_end2end_stop);
    PRINT(V_INFO, "Total execution time: %ld usec\n", total_end2end_micro);
#endif

#ifdef MEASURE_INFERENCE_TIME
    PRINT(V_INFO, "Single: number of individual images inference and read back: %d\n", total_images);
    PRINT(V_INFO, "Single: total inference time: %ld usec\n", total_inference_micro);
#endif

#ifdef MEASURE_GRAPH_INIT_TIME
    PRINT(V_INFO, "Graph initialization time: %ld usec\n", total_graph_init_micro);
    PRINT(V_INFO, "Graph size: %lu bytes\n", graphFileLength);
#endif

#ifdef MEASURE_H2D_TIME
    PRINT(V_INFO, "Single: total h2d time: %lu usec\n", total_h2d_micro);
#endif

#ifdef MEASURE_D2H_TIME
    PRINT(V_INFO, "Single: total d2h time: %lu usec\n", total_d2h_micro);
#endif

    PRINT(V_INFO, "Single: total data transfer time: %lu usec\n", total_d2h_micro + total_h2d_micro);

#ifdef MEASURE_INFERENCE_CALL
    PRINT(V_INFO, "Single: total inference call time: %lu usec\n", total_inference_call_micro);
#endif
    // 6. close session to release resources
    file_close(graphFile);
    kava_free(graphFileBuffer);
    kava_free(imageBuffer);
    njDone();

    return 0;
}


static int mvnc_stream(void) {
#ifdef MEASURE_END2END_TIME
    // End to end measurement
    struct timespec micro_end2end_start, micro_end2end_stop;
    long total_end2end_micro = 0;
#endif

#ifdef MEASURE_INFERENCE_TIME
    // inference measurement
    struct timespec micro_inference_start, micro_inference_stop;
    long total_inference_micro = 0;
#endif

#ifdef MEASURE_INFERENCE_CALL
    struct timespec micro_inference_call_start, micro_inference_call_stop;
    long total_inference_call_micro = 0;
#endif

#ifdef MEASURE_GRAPH_INIT_TIME
    // graph init measurement
    struct timespec micro_graph_init_start, micro_graph_init_stop;
    long total_graph_init_micro = 0;
#endif

#ifdef MEASURE_H2D_TIME
    // graph init measurement
    struct timespec micro_h2d_start, micro_h2d_stop;
    long total_h2d_micro = 0;
#endif

#ifdef MEASURE_D2H_TIME
    struct timespec micro_d2h_start, micro_d2h_stop;
    long total_d2h_micro = 0;
#endif


    ncStatus_t retCode;
    struct ncDeviceHandle_t *deviceHandle = NULL;
    int attempCounter = 10000;
    struct file *graphFile = NULL;
    struct file *image_file = NULL;
    struct ncGraphHandle_t *graphHandle;
    struct ncFifoHandle_t* inFifoHandle;
    struct ncFifoHandle_t* outFifoHandle;

    unsigned long graphFileLength;
    char *graphFileBuffer = NULL;
    unsigned long image_file_length;
    char *image_file_buffer = NULL;
    nj_result_t result;
    unsigned char *old_image_data = NULL;
    int old_image_data_length;
    unsigned int imageBufferLength;
    float* imageBuffer;
    int i;
    int passes;
    unsigned int outFifoElemSize;
    unsigned int optionSize;
    float* resultData = NULL;

#ifdef MEASURE_END2END_TIME
    getnstimeofday(&micro_end2end_start); 
#endif
    retCode = ncDeviceCreate(0, &deviceHandle);
    while (retCode != NC_OK) {
        if (attempCounter == 0) break;
        retCode = ncDeviceCreate(0, &deviceHandle);
        attempCounter--;
    }

    if (retCode != NC_OK) {
        printk(KERN_INFO "nvDeviceCreate failed.\n");
        return -1;
    }

    retCode = ncDeviceOpen(deviceHandle);
    attempCounter = 10000;
    while (retCode != NC_OK) {
        if (attempCounter == 0) break;
        retCode = ncDeviceOpen(deviceHandle);
        attempCounter--;
    }

    if (retCode != NC_OK) {
        printk(KERN_INFO "ncDeviceOpen failed.\n");
        return -1;
    }

    if (NULL == deviceHandle) {
        printk(KERN_INFO "ncDeviceOpen returned NULL handle\n");
        return -1;
    }

    graphFile = file_open(input_graph, O_RDONLY, 0);
    if (!graphFile) {
        pr_err("Can't open input_graph file.\n");
        return -1;
    }

    // Check file length
    graphFileLength = file_length(graphFile, input_graph);
    if (graphFileLength == 0) {
        printk(KERN_ALERT "graphFileLength failed\n");
        return -1;
    }

    graphFileBuffer = load_file(graphFile, graphFileLength);
    if (graphFileBuffer == NULL) {
        printk(KERN_ALERT "graphFileBuffer failed\n");
        return -1;
    }

#ifdef MEASURE_GRAPH_INIT_TIME
    getnstimeofday(&micro_graph_init_start); 
#endif

    retCode = ncGraphCreate("Inceptionv3-reshape", &graphHandle);
    if (retCode != NC_OK) {
        printk(KERN_ALERT "ncGraphCreate failed.\n");
        return -1;
    }

    retCode = ncGraphAllocateWithFifosEx(deviceHandle, graphHandle, graphFileBuffer, graphFileLength,
                                        &inFifoHandle, NC_FIFO_HOST_WO, total_images, NC_FIFO_FP32,
                                        &outFifoHandle, NC_FIFO_HOST_RO, total_images, NC_FIFO_FP32);
    if (retCode != NC_OK) {
        printk(KERN_ALERT "ncGraphAllocateWithFifosEx failed.\n");
        return -1;
    }

#ifdef MEASURE_GRAPH_INIT_TIME
    getnstimeofday(&micro_graph_init_stop);
    total_graph_init_micro += ELAPSED_TIME_MICRO_SEC(micro_graph_init_start, micro_graph_init_stop);
#endif

    // read tensor from image
    njInit();
    image_file = file_open(input_image, O_RDONLY, 0);
    if (!image_file) {
        pr_err("Can't open input_image file.\n");
        return -1;
    }
    image_file_length = file_length(image_file, input_image);
    if (image_file_length == 0) {
        printk(KERN_ALERT "graphFileLength failed\n");
        return -1;
    }
    image_file_buffer = load_file(image_file, image_file_length);
    if (image_file_buffer == NULL) {
        printk(KERN_ALERT "load image file failed\n");
        return -1;
    }

    result = njDecode(image_file_buffer, image_file_length);
    if (result != NJ_OK) {
        printk(KERN_ALERT "njDecode failed\n");
        return -1;
    } else {
        printk(KERN_ALERT "njDecode success.\n");
    }

    old_image_data = njGetImage();
    old_image_data_length = njGetImageSize();
   
    // convert uint8 to float
    imageBufferLength = old_image_data_length * sizeof(float);
    // printk(KERN_ALERT "nj.width is %d\n", nj.width);
    // printk(KERN_ALERT "nj.height is %d\n", nj.height);
    // printk(KERN_ALERT "nj.ncomp is %d\n", nj.ncomp);
    // printk(KERN_ALERT "imagebufferlength is %d\n", imageBufferLength);
    imageBuffer = kava_alloc(imageBufferLength);

    for (i = 0; i < old_image_data_length; ++i) {
        kernel_fpu_begin();
        imageBuffer[i] = (float)(old_image_data[i])/255.0;
        kernel_fpu_end();
    }

#ifdef MEASURE_INFERENCE_TIME
        getnstimeofday(&micro_inference_start);
#endif

    for (passes = 0; passes < total_images; passes++) {
#ifdef MEASURE_H2D_TIME
        getnstimeofday(&micro_h2d_start);
#endif
        // 4. Run the image through the model
        retCode = ncFifoWriteElem(inFifoHandle, imageBuffer, &imageBufferLength, NULL);
        if (retCode != NC_OK) {
            printk(KERN_ALERT "ncFifoWriteElem failed.\n");
            return -1;
        }

#ifdef MEASURE_H2D_TIME
        getnstimeofday(&micro_h2d_stop);
        total_h2d_micro += ELAPSED_TIME_MICRO_SEC(micro_h2d_start,
                micro_h2d_stop);
#endif

#ifdef MEASURE_INFERENCE_CALL
        getnstimeofday(&micro_inference_call_start);
#endif  

        retCode = ncGraphQueueInference(graphHandle, &inFifoHandle, 1,
                                &outFifoHandle, 1);
        if (retCode != NC_OK) {
            printk(KERN_ALERT "ncGraphQueueInference failed.\n");
            return -1;
        }

#ifdef MEASURE_INFERENCE_CALL
        getnstimeofday(&micro_inference_call_stop);
        total_inference_call_micro += ELAPSED_TIME_MICRO_SEC(micro_inference_call_start,
                micro_inference_call_stop);
#endif
        outFifoElemSize = 0;
        optionSize = sizeof(outFifoElemSize);
        retCode = ncFifoGetOption(outFifoHandle, NC_RO_FIFO_ELEMENT_DATA_SIZE,
                &outFifoElemSize, &optionSize);
        if (retCode != NC_OK) {
            printk(KERN_ALERT "ncFifoGetOption failed.\n");
            return -1;
        }

        // Get the output tensor
        resultData = (float*) kava_alloc(outFifoElemSize);

#ifdef MEASURE_D2H_TIME
        getnstimeofday(&micro_d2h_start); 
#endif
        // We don't support userParam...
        retCode = ncFifoReadElem(outFifoHandle, (void*)resultData, &outFifoElemSize, NULL);
        if (retCode != NC_OK) {
            printk(KERN_ALERT "ncFifoReadElem failed.\n");
            return -1;
        }

#ifdef MEASURE_D2H_TIME
        getnstimeofday(&micro_d2h_stop);
        total_d2h_micro += ELAPSED_TIME_MICRO_SEC(micro_d2h_start, micro_d2h_stop);
#endif

        // This may be commented out
        kava_free(resultData);
    }

#ifdef MEASURE_INFERENCE_TIME
    getnstimeofday(&micro_inference_stop); 
    total_inference_micro += ELAPSED_TIME_MICRO_SEC(micro_inference_start, micro_inference_stop);
#endif

    // Clean up the FIFOs
    ncFifoDestroy(&inFifoHandle);
    ncFifoDestroy(&outFifoHandle);

    // Clean up the graph
    ncGraphDestroy(&graphHandle);

    // Close and clean up the device
    ncDeviceClose(deviceHandle);
    ncDeviceDestroy(&deviceHandle);

#ifdef MEASURE_END2END_TIME
    getnstimeofday(&micro_end2end_stop);
    total_end2end_micro += ELAPSED_TIME_MICRO_SEC(micro_end2end_start, micro_end2end_stop);
    PRINT(V_INFO, "Total execution time: %ld usec\n", total_end2end_micro);
#endif

#ifdef MEASURE_INFERENCE_TIME
    PRINT(V_INFO, "Streaming: number of individual images inference and read back: %d\n", total_images);
    PRINT(V_INFO, "Streaming: total inference time: %ld usec\n", total_inference_micro);
#endif

#ifdef MEASURE_GRAPH_INIT_TIME
    PRINT(V_INFO, "Graph initialization time: %ld usec\n", total_graph_init_micro);
    PRINT(V_INFO, "Graph size: %lu bytes\n", graphFileLength);
#endif

#ifdef MEASURE_H2D_TIME
    PRINT(V_INFO, "Streaming: total h2d time: %lu usec\n", total_h2d_micro);
#endif

#ifdef MEASURE_D2H_TIME
    PRINT(V_INFO, "Streaming: total d2h time: %lu usec\n", total_d2h_micro);
#endif

    PRINT(V_INFO, "Streaming: total data transfer time: %lu usec\n", total_d2h_micro + total_h2d_micro);

#ifdef MEASURE_INFERENCE_CALL
    PRINT(V_INFO, "Streaming: total inference call time: %lu usec\n", total_inference_call_micro);
#endif
    // 6. close session to release resources
    file_close(graphFile);
    file_close(image_file);
    kava_free(graphFileBuffer);
    kava_free(imageBuffer);
    njDone();

    return 0;
}

static int mvnc_batch(void) {
#ifdef MEASURE_END2END_TIME
    // End to end measurement
    struct timespec micro_end2end_start, micro_end2end_stop;
    long total_end2end_micro = 0;
#endif

#ifdef MEASURE_INFERENCE_TIME
    // inference measurement
    struct timespec micro_inference_start, micro_inference_stop;
    long total_inference_micro = 0;
#endif

#ifdef MEASURE_INFERENCE_CALL
    struct timespec micro_inference_call_start, micro_inference_call_stop;
    long total_inference_call_micro = 0;
#endif

#ifdef MEASURE_GRAPH_INIT_TIME
    // graph init measurement
    struct timespec micro_graph_init_start, micro_graph_init_stop;
    long total_graph_init_micro = 0;
#endif

#ifdef MEASURE_H2D_TIME
    // graph init measurement
    struct timespec micro_h2d_start, micro_h2d_stop;
    long total_h2d_micro = 0;
#endif

#ifdef MEASURE_D2H_TIME
    struct timespec micro_d2h_start, micro_d2h_stop;
    long total_d2h_micro = 0;
#endif


    ncStatus_t retCode;
    struct ncDeviceHandle_t *deviceHandle = NULL;
    int attempCounter = 10000;
    struct file *graphFile = NULL;
    struct file *image_file = NULL;
    struct ncGraphHandle_t *graphHandle;
    struct ncFifoHandle_t* inFifoHandle;
    struct ncFifoHandle_t* outFifoHandle;

    unsigned long graphFileLength;
    char *graphFileBuffer = NULL;
    unsigned long image_file_length;
    char *image_file_buffer = NULL;
    nj_result_t result;
    unsigned char *old_image_data = NULL;
    int old_image_data_length;
    unsigned int imageBufferLength;
    float* imageBuffer;
    int i;
    int passes;
    unsigned int outFifoElemSize;
    unsigned int optionSize;
    float* resultData = NULL;

#ifdef MEASURE_END2END_TIME
    getnstimeofday(&micro_end2end_start); 
#endif
    retCode = ncDeviceCreate(0, &deviceHandle);
    while (retCode != NC_OK) {
        if (attempCounter == 0) break;
        retCode = ncDeviceCreate(0, &deviceHandle);
        attempCounter--;
    }

    if (retCode != NC_OK) {
        printk(KERN_INFO "nvDeviceCreate failed.\n");
        return -1;
    }

    retCode = ncDeviceOpen(deviceHandle);
    attempCounter = 10000;
    while (retCode != NC_OK) {
        if (attempCounter == 0) break;
        retCode = ncDeviceOpen(deviceHandle);
        attempCounter--;
    }

    if (retCode != NC_OK) {
        printk(KERN_INFO "ncDeviceOpen failed.\n");
        return -1;
    }

    if (NULL == deviceHandle) {
        printk(KERN_INFO "ncDeviceOpen returned NULL handle\n");
        return -1;
    }

    graphFile = file_open(input_graph, O_RDONLY, 0);
    if (!graphFile) {
        pr_err("Can't open input_graph file.\n");
        return -1;
    }

    // Check file length
    graphFileLength = file_length(graphFile, input_graph);
    if (graphFileLength == 0) {
        printk(KERN_ALERT "graphFileLength failed\n");
        return -1;
    }

    graphFileBuffer = load_file(graphFile, graphFileLength);
    if (graphFileBuffer == NULL) {
        printk(KERN_ALERT "graphFileBuffer failed\n");
        return -1;
    }

#ifdef MEASURE_GRAPH_INIT_TIME
    getnstimeofday(&micro_graph_init_start); 
#endif

    retCode = ncGraphCreate("Inceptionv3-reshape", &graphHandle);
    if (retCode != NC_OK) {
        printk(KERN_ALERT "ncGraphCreate failed.\n");
        return -1;
    }

    retCode = ncGraphAllocateWithFifosEx(deviceHandle, graphHandle, graphFileBuffer, graphFileLength,
                                        &inFifoHandle, NC_FIFO_HOST_WO, total_images, NC_FIFO_FP32,
                                        &outFifoHandle, NC_FIFO_HOST_RO, total_images, NC_FIFO_FP32);
    if (retCode != NC_OK) {
        printk(KERN_ALERT "ncGraphAllocateWithFifosEx failed.\n");
        return -1;
    }

#ifdef MEASURE_GRAPH_INIT_TIME
    getnstimeofday(&micro_graph_init_stop);
    total_graph_init_micro += ELAPSED_TIME_MICRO_SEC(micro_graph_init_start, micro_graph_init_stop);
#endif

    // read tensor from image
    njInit();
    image_file = file_open(input_image, O_RDONLY, 0);
    if (!image_file) {
        pr_err("Can't open input_image file.\n");
        return -1;
    }
    image_file_length = file_length(image_file, input_image);
    if (image_file_length == 0) {
        printk(KERN_ALERT "graphFileLength failed\n");
        return -1;
    }
    image_file_buffer = load_file(image_file, image_file_length);
    if (image_file_buffer == NULL) {
        printk(KERN_ALERT "load image file failed\n");
        return -1;
    }

    result = njDecode(image_file_buffer, image_file_length);
    if (result != NJ_OK) {
        printk(KERN_ALERT "njDecode failed\n");
        return -1;
    } else {
        printk(KERN_ALERT "njDecode success.\n");
    }

    old_image_data = njGetImage();
    old_image_data_length = njGetImageSize();
   
    // convert uint8 to float
    imageBufferLength = old_image_data_length * sizeof(float);
    // printk(KERN_ALERT "nj.width is %d\n", nj.width);
    // printk(KERN_ALERT "nj.height is %d\n", nj.height);
    // printk(KERN_ALERT "nj.ncomp is %d\n", nj.ncomp);
    // printk(KERN_ALERT "imagebufferlength is %d\n", imageBufferLength);
    imageBuffer = kava_alloc(imageBufferLength);

    for (i = 0; i < old_image_data_length; ++i) {
        kernel_fpu_begin();
        imageBuffer[i] = (float)(old_image_data[i])/255.0;
        kernel_fpu_end();
    }

#ifdef MEASURE_INFERENCE_TIME
    getnstimeofday(&micro_inference_start);
#endif
#ifdef MEASURE_H2D_TIME
    getnstimeofday(&micro_h2d_start);
#endif
    for (passes = 0; passes < total_images; passes++) {
        // 4. Run the image through the model
        retCode = ncFifoWriteElem(inFifoHandle, imageBuffer, &imageBufferLength, NULL);
        if (retCode != NC_OK) {
            printk(KERN_ALERT "ncFifoWriteElem failed.\n");
            return -1;
        }
    }
#ifdef MEASURE_H2D_TIME
    getnstimeofday(&micro_h2d_stop);
    total_h2d_micro += ELAPSED_TIME_MICRO_SEC(micro_h2d_start,
                micro_h2d_stop);
#endif

#ifdef MEASURE_INFERENCE_CALL
        getnstimeofday(&micro_inference_call_start);
#endif
    for (passes = 0; passes < total_images; passes++) {
        retCode = ncGraphQueueInference(graphHandle, &inFifoHandle, 1,
                                &outFifoHandle, 1);
        if (retCode != NC_OK) {
            printk(KERN_ALERT "ncGraphQueueInference failed.\n");
            return -1;
        }
    }
#ifdef MEASURE_INFERENCE_CALL
    getnstimeofday(&micro_inference_call_stop);
    total_inference_call_micro += ELAPSED_TIME_MICRO_SEC(micro_inference_call_start,
                micro_inference_call_stop);
#endif
    for (passes = 0; passes < total_images; passes++) {
        outFifoElemSize = 0;
        optionSize = sizeof(outFifoElemSize);
        retCode = ncFifoGetOption(outFifoHandle, NC_RO_FIFO_ELEMENT_DATA_SIZE,
            &outFifoElemSize, &optionSize);
        if (retCode != NC_OK) {
            printk(KERN_ALERT "ncFifoGetOption failed.\n");
            return -1;
        }

        // Get the output tensor
        resultData = (float*) kava_alloc(outFifoElemSize);

#ifdef MEASURE_D2H_TIME
        getnstimeofday(&micro_d2h_start); 
#endif
        // We don't support userParam...
        retCode = ncFifoReadElem(outFifoHandle, (void*)resultData, &outFifoElemSize, NULL);
        if (retCode != NC_OK) {
            printk(KERN_ALERT "ncFifoReadElem failed.\n");
            return -1;
        }

#ifdef MEASURE_D2H_TIME
        getnstimeofday(&micro_d2h_stop);
        total_d2h_micro += ELAPSED_TIME_MICRO_SEC(micro_d2h_start, micro_d2h_stop);
#endif

        // This may be commented out
        kava_free(resultData);
    }

#ifdef MEASURE_INFERENCE_TIME
    getnstimeofday(&micro_inference_stop); 
    total_inference_micro += ELAPSED_TIME_MICRO_SEC(micro_inference_start, micro_inference_stop);
#endif

    // Clean up the FIFOs
    ncFifoDestroy(&inFifoHandle);
    ncFifoDestroy(&outFifoHandle);

    // Clean up the graph
    ncGraphDestroy(&graphHandle);

    // Close and clean up the device
    ncDeviceClose(deviceHandle);
    ncDeviceDestroy(&deviceHandle);

#ifdef MEASURE_END2END_TIME
    getnstimeofday(&micro_end2end_stop);
    total_end2end_micro += ELAPSED_TIME_MICRO_SEC(micro_end2end_start, micro_end2end_stop);
    PRINT(V_INFO, "Total execution time: %ld usec\n", total_end2end_micro);
#endif

#ifdef MEASURE_INFERENCE_TIME
    PRINT(V_INFO, "Batch: number of individual images inference and read back: %d\n", total_images);
    PRINT(V_INFO, "Batch: total inference time: %ld usec\n", total_inference_micro);
#endif

#ifdef MEASURE_GRAPH_INIT_TIME
    PRINT(V_INFO, "Graph initialization time: %ld usec\n", total_graph_init_micro);
    PRINT(V_INFO, "Graph size: %lu bytes\n", graphFileLength);
#endif

#ifdef MEASURE_H2D_TIME
    PRINT(V_INFO, "Batch: total h2d time: %lu usec\n", total_h2d_micro);
#endif

#ifdef MEASURE_D2H_TIME
    PRINT(V_INFO, "Batch: total d2h time: %lu usec\n", total_d2h_micro);
#endif

    PRINT(V_INFO, "Batch: total data transfer time: %lu usec\n", total_d2h_micro + total_h2d_micro);

#ifdef MEASURE_INFERENCE_CALL
    PRINT(V_INFO, "Batch: total inference call time: %lu usec\n", total_inference_call_micro);
#endif
    // 6. close session to release resources
    file_close(graphFile);
    file_close(image_file);
    kava_free(graphFileBuffer);
    kava_free(imageBuffer);
    njDone();

    return 0;
}

static int __init mvnc_inception_init(void) {
    if (batch_mode == 0) {
        return mvnc_stream();
    } else if (batch_mode == 1) {
        return mvnc_batch();
    } else if (batch_mode == 2) {
        return mvnc_single();
    } else {
        PRINT(V_ERROR, "Known img processing mode: %d\n", batch_mode);
        return -1;
    }
}

static void __exit mvnc_inception_exit(void) {
    printk(KERN_INFO "MVNC inception finished\n");
}

module_init(mvnc_inception_init);
module_exit(mvnc_inception_exit);
