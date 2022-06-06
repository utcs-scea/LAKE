#define pr_fmt(fmt) "%s:%d:: " fmt, __func__, __LINE__
#include <linux/types.h>

#include "util.h"
#include "cuda_kava.h"

CUresult cuda_driver_api_init(CUcontext *pctx, CUmodule *pmod, const char *f)
{
	CUresult res;
	CUdevice dev;

	res = cuInit(0);
	if (res != CUDA_SUCCESS) {
		pr_err("cuInit failed: res = %lu\n", (unsigned long)res);
		return res;
	}

	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		pr_err("cuDeviceGet failed: res = %lu\n", (unsigned long)res);
		return res;
	}

	res = cuCtxCreate(pctx, 0, dev);
	if (res != CUDA_SUCCESS) {
		pr_err("cuCtxCreate failed: res = %lu\n", (unsigned long)res);
		return res;
	}

	res = cuModuleLoad(pmod, f);
	if (res != CUDA_SUCCESS) {
		pr_err("cuModuleLoad() failed\n");
		cuCtxDestroy(*pctx);
		return res;
	}

	return CUDA_SUCCESS;
}

CUresult cuda_driver_api_exit(CUcontext ctx, CUmodule mod)
{
	CUresult res;

	res = cuModuleUnload(mod);
	if (res != CUDA_SUCCESS) {
		pr_err("cuModuleUnload failed: res = %lu\n", (unsigned long)res);
		return res;
	}

	res = cuCtxDestroy(ctx);
	if (res != CUDA_SUCCESS) {
		pr_err("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
		return res;
	}

	return CUDA_SUCCESS;
}

void probe_time_start(struct timestamp *ts)
{
    getnstimeofday(&ts->start);
}

long probe_time_end(struct timestamp *ts)
{
    struct timespec tv;
    getnstimeofday(&ts->end);
	tvsub(&ts->end, &ts->start, &tv);
	return (tv.tv_sec * 1000000000 + tv.tv_nsec);
}
