//    2006.03   Rob Janiczek
//        --creation of prototype version
//    2006.03   Drew Gilliam
//        --rewriting of prototype version into current version
//        --got rid of multiple function calls, all code in a  
//         single function (for speed)
//        --code cleanup & commenting
//        --code optimization efforts   
//    2006.04   Drew Gilliam
//        --added diffusion coefficent saturation on [0,1]
//		2009.12 Lukasz G. Szafaryn
//		-- reading from image, command line inputs
//		2010.01 Lukasz G. Szafaryn
//		--comments
//    2011.12 Shinpei Kato
//        --modified to use Driver API
//    2020.01 Hangchen Yu
//        --ported to Linux kernel

#define pr_fmt(fmt) "%s:%d:: " fmt, __func__, __LINE__
#include <linux/module.h>
#include <linux/delay.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/random.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>

#include "cuda_kava.h"
#include "../util/util.h"
#include "srad.h"

#include "graphics.c"
#include "resize.c"
#include "timer.c"

/* (in library path specified to compiler)	needed by for device functions */
#include "device.c"

static int niter = 0;
static int _lambda = 2; // 1/lambda (0.5)
static int Nr = 0;
static int Nc = 0;
static char *cubin_p = NULL;
static char *input_f = NULL;
static char *output_f = NULL;
module_param(niter, int, S_IRUSR);
MODULE_PARM_DESC(niter, "Number of iterations");
module_param(_lambda, int, S_IRUSR);
MODULE_PARM_DESC(_lambda, "Reciprocal of lambda (1/lambda), the step size");
module_param(Nr, int, S_IRUSR);
MODULE_PARM_DESC(Nr, "Number of rows in image");
module_param(Nc, int, S_IRUSR);
MODULE_PARM_DESC(Nc, "Number of columns in image");
module_param(cubin_p, charp, 0000);
MODULE_PARM_DESC(cubin_p, "CUDA binary path");
module_param(input_f, charp, 0000);
MODULE_PARM_DESC(input_f, "Input image file path");
module_param(output_f, charp, 0000);
MODULE_PARM_DESC(output_f, "Output image file path");

CUresult extract_launch
(CUmodule mod, int gdx, int gdy, int bdx, int bdy, long Ne, CUdeviceptr d_I)
{
	CUfunction f;
	CUresult res;
	void* param[] = {&Ne, &d_I, NULL};

	res = cuModuleGetFunction(&f, mod, "_Z7extractlPi");
	if (res != CUDA_SUCCESS) {
		pr_err("cuModuleGetFunction(extract) failed: res = %u\n", res);
		return res;
	}

	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**)param, 0);
	if (res != CUDA_SUCCESS) {
		pr_err("cuLaunchKernel(extract) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

CUresult prepare_launch
(CUmodule mod, int gdx, int gdy, int bdx, int bdy, long Ne, CUdeviceptr d_I,
 CUdeviceptr d_sums, CUdeviceptr d_sums2)
{
	CUfunction f;
	CUresult res;
	void* param[] = {&Ne, &d_I, &d_sums, &d_sums2, NULL};

	/* get functions. */
	res = cuModuleGetFunction(&f, mod, "_Z7preparelPiS_S_");
	if (res != CUDA_SUCCESS) {
		pr_err("cuModuleGetFunction(prepare) failed: res = %u\n", res);
		return res;
	}

	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**)param, 0);
	if (res != CUDA_SUCCESS) {
		pr_err("cuLaunchKernel(prepare) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

CUresult reduce_launch
(CUmodule mod, int gdx, int gdy, int bdx, int bdy, long Ne, int no, int mul, 
 CUdeviceptr d_sums, CUdeviceptr d_sums2)
{
	CUfunction f;
	CUresult res;
	void* param[] = {&Ne, &no, &mul, &d_sums, &d_sums2, NULL};

	/* get functions. */
	res = cuModuleGetFunction(&f, mod, "_Z6reduceliiPiS_");
	if (res != CUDA_SUCCESS) {
		pr_err("cuModuleGetFunction(reduce) failed: res = %u\n", res);
		return res;
	}
	
	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**)param, 0);
	if (res != CUDA_SUCCESS) {
		pr_err("cuLaunchKernel(reduce) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

CUresult srad_launch
(CUmodule mod, int gdx, int gdy, int bdx, int bdy, int _lambda, int Nr, int Nc, 
 long Ne, CUdeviceptr d_iN, CUdeviceptr d_iS, CUdeviceptr d_jE, 
 CUdeviceptr d_jW, CUdeviceptr d_dN, CUdeviceptr d_dS, CUdeviceptr d_dE, 
 CUdeviceptr d_dW, long q0sqr, CUdeviceptr d_c, CUdeviceptr d_I)
{
	CUfunction f;
	CUresult res;
	void* param[] = {&_lambda, &Nr, &Nc, &Ne, &d_iN, &d_iS, &d_jE, &d_jW, &d_dN,
					 &d_dS, &d_dE, &d_dW, &q0sqr, &d_c, &d_I, NULL};

	/* get functions. */
	res = cuModuleGetFunction(&f, mod, "_Z4sradiiilPiS_S_S_S_S_S_S_lS_S_");
	if (res != CUDA_SUCCESS) {
		pr_err("cuModuleGetFunction(srad) failed: res = %u\n", res);
		return res;
	}
	
	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**)param, 0);
	if (res != CUDA_SUCCESS) {
		pr_err("cuLaunchKernel(srad) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

CUresult srad2_launch
(CUmodule mod, int gdx, int gdy, int bdx, int bdy, int _lambda, int Nr, int Nc, 
 long Ne, CUdeviceptr d_iN, CUdeviceptr d_iS, CUdeviceptr d_jE,  
 CUdeviceptr d_jW, CUdeviceptr d_dN, CUdeviceptr d_dS, CUdeviceptr d_dE, 
 CUdeviceptr d_dW, CUdeviceptr d_c, CUdeviceptr d_I)
{
	CUfunction f;
	CUresult res;
	void* param[] = {&_lambda, &Nr, &Nc, &Ne, &d_iN, &d_iS, &d_jE, &d_jW, &d_dN,
					 &d_dS, &d_dE, &d_dW, &d_c, &d_I, NULL};

	/* get functions. */
	res = cuModuleGetFunction(&f, mod, "_Z5srad2iiilPiS_S_S_S_S_S_S_S_S_");
	if (res != CUDA_SUCCESS) {
		pr_err("cuModuleGetFunction(srad2) failed: res = %u\n", res);
		return res;
	}
	
	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**)param, 0);
	if (res != CUDA_SUCCESS) {
		pr_err("cuLaunchKernel(srad2) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

CUresult compress_launch
(CUmodule mod, int gdx, int gdy, int bdx, int bdy, long Ne, CUdeviceptr d_I)
{
	CUfunction f;
	CUresult res;
	void* param[] = {&Ne, &d_I, NULL};

	res = cuModuleGetFunction(&f, mod, "_Z8compresslPi");
	if (res != CUDA_SUCCESS) {
		pr_err("cuModuleGetFunction(compress) failed: res = %u\n", res);
		return res;
	}
	
	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**)param, 0);
	if (res != CUDA_SUCCESS) {
		pr_err("cuLaunchKernel(compress) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

static int __init drv_init(void)
{
	/* time */
    struct timestamp ts_init, ts_total, ts_memalloc, ts_h2d, ts_d2h, ts_kernel, ts_close;
    long init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
          d2h_time = 0, close_time = 0, total_time = 0;

	/* inputs image, input paramenters */
	int* image_ori; /* originalinput image */
	int image_ori_rows;
	int image_ori_cols;
	long image_ori_elem;

	/* inputs image, input paramenters */
	int* image; /* input image */
	long Ne;

	/* size of IMAGE */
	int r1, r2, c1, c2; /* row/col coordinates of uniform ROI */
	long NeROI;	/* ROI nbr of elements */

	/* surrounding pixel indicies */
	int *iN, *iS, *jE, *jW;	
	
	/* counters */
	int iter; /* primary loop */
	long i, j; /* image row/col */

	/* memory sizes */
	int mem_size_i;
	int mem_size_j;
	int mem_size_single;

	/*******************************************************
	 * 	GPU VARIABLES
	 ******************************************************/
	/* CUDA kernel execution parameters */
	int bdx, bdy;
	int x;
	int gdx, gdy;
	int gdx2, gdy2;

	/*  memory sizes */
	int mem_size; /* matrix memory size */

	/* HOST */
	int no;
	int mul;
	long total;
	long total2;
	long meanROI;
	long meanROI2;
	long varROI;
	long q0sqr;

	/* DEVICE */
	CUdeviceptr d_sums;	/* partial sum */
	CUdeviceptr d_sums2;
	CUdeviceptr d_iN;
	CUdeviceptr d_iS;
	CUdeviceptr d_jE;
	CUdeviceptr d_jW;
	CUdeviceptr d_dN; 
	CUdeviceptr d_dS; 
	CUdeviceptr d_dW; 
	CUdeviceptr d_dE;
	CUdeviceptr d_I; /* input IMAGE on DEVICE */
	CUdeviceptr d_c;

	CUcontext ctx;
	CUmodule mod;
	CUresult res;

    char cubin_fn[128];

    pr_info("load cuda srad_v1 module\n");

	/*******************************************************
	 * 	GET INPUT PARAMETERS
	 ******************************************************/
	if (niter == 0 || _lambda == 0 || Nr == 0 || Nc == 0 || !cubin_p) {
		pr_err("ERROR: wrong number of arguments\n");
		return 0;
	}

	/*******************************************************
	 * READ IMAGE (SIZE OF IMAGE HAS TO BE KNOWN)
	 ******************************************************/
	/* read image */
	image_ori_rows = 502;
	image_ori_cols = 458;
	image_ori_elem = image_ori_rows * image_ori_cols;

	image_ori = (int*) vmalloc(sizeof(int) * image_ori_elem);

    if (input_f)
        read_graphics(input_f, image_ori, image_ori_rows, image_ori_cols, 1);

	/*******************************************************
	 * RESIZE IMAGE (ASSUMING COLUMN MAJOR STORAGE OF image_orig)
	 ******************************************************/
	Ne = Nr * Nc;

	image = (int*) vmalloc(sizeof(int) * Ne);

	resize(image_ori, image_ori_rows, image_ori_cols, image, Nr, Nc, 1);

	/*******************************************************
	 * SETUP
	 ******************************************************/
	r1 = 0;	/* top row index of ROI */
	r2 = Nr - 1; /* bottom row index of ROI */
	c1 = 0;	/* left column index of ROI */
	c2 = Nc - 1; /* right column index of ROI */

	/* ROI image size */
	NeROI = (r2 - r1 + 1) * (c2 - c1 + 1); /* # of elements in ROI, ROI size */

	/* allocate variables for surrounding pixels */
	mem_size_i = sizeof(int) * Nr;
	iN = (int *) vmalloc(mem_size_i); /* north surrounding element */
	iS = (int *) vmalloc(mem_size_i); /* south surrounding element */
	mem_size_j = sizeof(int) * Nc;
	jW = (int *) vmalloc(mem_size_j); /* west surrounding element */
	jE = (int *) vmalloc(mem_size_j); /* east surrounding element */

	/* N/S/W/E indices of surrounding pixels (every element of IMAGE) */
	for (i = 0; i < Nr; i++) {
		iN[i] = i - 1; /* holds index of IMAGE row above */
		iS[i] = i + 1; /* holds index of IMAGE row below */
	}
	for (j = 0; j < Nc; j++) {
		jW[j] = j - 1; /* holds index of IMAGE column on the left */
		jE[j] = j + 1; /* holds index of IMAGE column on the right */
	}

	/* N/S/W/E boundary conditions, 
	   fix surrounding indices outside boundary of image */
	iN[0] = 0; /* changes IMAGE top row index from -1 to 0 */
	iS[Nr - 1] = Nr - 1; /* changes IMAGE bottom row index from Nr to Nr-1 */
	jW[0] = 0; /* changes IMAGE leftmost column index from -1 to 0 */
	jE[Nc - 1] = Nc - 1; /* changes IMAGE rightmost col idx from Nc to Nc-1 */

	/*******************************************************
	 * GPU SETUP
	 ******************************************************/

	/* call our common CUDA initialization utility function. */
    strcpy(cubin_fn, cubin_p);
    strcat(cubin_fn, "/srad.cubin");

    probe_time_start(&ts_total);
    probe_time_start(&ts_init);

	res = cuda_driver_api_init(&ctx, &mod, cubin_fn);
	if (res != CUDA_SUCCESS) {
		pr_err("cuda_driver_api_init failed: res = %u\n", res);
		return -1;
	}

    init_time = probe_time_end(&ts_init);
    probe_time_start(&ts_memalloc);

	/* allocate memory for entire IMAGE on DEVICE */
	mem_size = sizeof(int) * Ne;	/* size of input IMAGE */
	res = cuMemAlloc(&d_I, mem_size);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}

	/* allocate memory for coordinates on DEVICE */
	res = cuMemAlloc(&d_iN, mem_size_i);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_iS, mem_size_i);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_jE, mem_size_j);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_jW, mem_size_j);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}

	/* allocate memory for partial sums on DEVICE */
	res = cuMemAlloc(&d_sums, mem_size);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_sums2, mem_size);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}

	/* allocate memory for derivatives */
	res = cuMemAlloc(&d_dN, mem_size);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_dS, mem_size);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_dW, mem_size);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_dE, mem_size);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}

	/* allocate memory for coefficient on DEVICE */
	res = cuMemAlloc(&d_c, mem_size);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}

    mem_alloc_time = probe_time_end(&ts_memalloc);

	/*******************************************************
	 * COPY DATA TO DEVICE 
	 ******************************************************/
    probe_time_start(&ts_h2d);

	res = cuMemcpyHtoD(d_iN, iN, mem_size_i);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}
	res = cuMemcpyHtoD(d_iS, iS, mem_size_i);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}
	res = cuMemcpyHtoD(d_jE, jE, mem_size_j);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}
	res = cuMemcpyHtoD(d_jW, jW, mem_size_j);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}

	/* checkCUDAError("setup"); */

	/*******************************************************
	 * KERNEL EXECUTION PARAMETERS
	 ******************************************************/

	/* all kernels operating on entire matrix */
	bdx = NUMBER_THREADS; /* define # of threads in the block */
	bdy = 1;
	x = Ne / bdx;
	/* compensate for division remainder above by adding one grid */
	if (Ne % bdx != 0) {
		x = x + 1;
	}
	gdx = x; /* define # of blocks in the grid */
	gdy = 1;

	/*******************************************************
	 * COPY INPUT TO GPU
	 ******************************************************/

	res = cuMemcpyHtoD(d_I, image, mem_size);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}

    h2d_time = probe_time_end(&ts_h2d);

	/*******************************************************
	 * SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
	 ******************************************************/
    probe_time_start(&ts_kernel);

	res = extract_launch(mod, gdx, gdy, bdx, bdy, Ne, d_I);
	if (res != CUDA_SUCCESS) {
		pr_err("extract_launch failed: res = %u\n", res);
		return -1;
	}

    cuCtxSynchronize();
    kernel_time = probe_time_end(&ts_kernel);

	/* checkCUDAError("extract"); */

	/*******************************************************
	 * COMPUTATION
	 ******************************************************/
	/* execute main loop:
	   do for # of iterations input parameter */
	for (iter = 0; iter < niter; iter++) {
		/* pr_err("%d ", iter); */
		/* fflush(NULL); */

        probe_time_start(&ts_kernel);
		/* execute square kernel */
		res = prepare_launch(mod, gdx, gdy, bdx, bdy, Ne, d_I, d_sums, d_sums2);
		if (res != CUDA_SUCCESS) {
			pr_err("prepare_launch failed: res = %u\n", res);
			return -1;
		}

		/* checkCUDAError("prepare"); */

		/* performs subsequent reductions of sums */
		gdx2 = gdx; /* original number of blocks */
		gdy2 = gdy;
		no = Ne; /* original number of sum elements */
		mul = 1; /* original multiplier */

		while (gdx2 != 0) {
			/* checkCUDAError("before reduce"); */
			/* run kernel */
			res = reduce_launch(mod, gdx2, gdy2, bdx, bdy, Ne, no, mul, 
								d_sums, d_sums2);
			if (res != CUDA_SUCCESS) {
				pr_err("reduce_launch failed: res = %u\n", res);
				return -1;
			}

			/* checkCUDAError("reduce"); */

			/* update execution parameters */
			no = gdx2; /* get current number of elements */
			if (gdx2 == 1) {
				gdx2 = 0;
			}
			else {
				mul = mul * NUMBER_THREADS;	/* update the increment */
				x = gdx2 / bdx;	/* # of blocks */
				/* compensate for division remainder above by adding one grid */
				if (gdx2 % bdx != 0) {
					x = x + 1;
				}
				gdx2 = x;
				gdy2 = 1;
			}
			/* checkCUDAError("after reduce"); */

		}

		/* checkCUDAError("before copy sum"); */
        cuCtxSynchronize();
        kernel_time += probe_time_end(&ts_kernel);

		/* copy total sums to HOST */
		mem_size_single = sizeof(int) * 1;
        probe_time_start(&ts_d2h);
		res = cuMemcpyDtoH(&total, d_sums, mem_size_single);
		if (res != CUDA_SUCCESS) {
			pr_err("cuMemcpyDtoH failed: res = %u\n", res);
			return -1;
		}
		res = cuMemcpyDtoH(&total2, d_sums2, mem_size_single);
		if (res != CUDA_SUCCESS) {
			pr_err("cuMemcpyDtoH failed: res = %u\n", res);
			return -1;
		}
        d2h_time += probe_time_end(&ts_d2h);

		/* checkCUDAError("copy sum"); */
		/* calculate statistics */
		meanROI	= total * 1000 / NeROI; /* mean (avg.) value of element in ROI */
		meanROI2 = meanROI * meanROI / 1000;
		varROI = (total2 *1000 / NeROI) - meanROI2; /* variance of ROI */
		q0sqr = varROI * 1000 / meanROI2; /* standard deviation of ROI */

        probe_time_start(&ts_kernel);
		/* execute srad kernel */
		res = srad_launch(mod, gdx, gdy, bdx, bdy,
						  _lambda, // SRAD coefficient 
						  Nr, // # of rows in input image
						  Nc, // # of columns in input image
						  Ne, // # of elements in input image
						  d_iN,	// indices of North surrounding pixels
						  d_iS, // indices of South surrounding pixels
						  d_jE, // indices of East surrounding pixels
						  d_jW,	// indices of West surrounding pixels
						  d_dN,	// North derivative
						  d_dS,	// South derivative
						  d_dE,	// East derivative
						  d_dW,	// West derivative
						  q0sqr, // standard deviation of ROI 
						  d_c, // diffusion coefficient
						  d_I // output image
			);
		if (res != CUDA_SUCCESS) {
			pr_err("srad_launch failed: res = %u\n", res);
			return -1;
		}
		/* checkCUDAError("srad"); */
		
		/* execute srad2 kernel */
		res = srad2_launch(mod, gdx, gdy, bdx, bdy,
						   _lambda,	// SRAD coefficient 
						   Nr, // # of rows in input image
						   Nc, // # of columns in input image
						   Ne, // # of elements in input image
						   d_iN, // indices of North surrounding pixels
						   d_iS, // indices of South surrounding pixels
						   d_jE, // indices of East surrounding pixels
						   d_jW, // indices of West surrounding pixels
						   d_dN, // North derivative
						   d_dS, // South derivative
						   d_dE, // East derivative
						   d_dW, // West derivative
						   d_c, // diffusion coefficient
						   d_I // output image
			);
		if (res != CUDA_SUCCESS) {
			pr_err("srad2_launch failed: res = %u\n", res);
			return -1;
		}
		/* checkCUDAError("srad2"); */

        cuCtxSynchronize();
        kernel_time += probe_time_end(&ts_kernel);
	}

	/*******************************************************
	 * SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
	 ******************************************************/

    probe_time_start(&ts_kernel);

	res = compress_launch(mod, gdx, gdy, bdx, bdy, Ne, d_I);
	if (res != CUDA_SUCCESS) {
		pr_err("compress_launch failed: res = %u\n", res);
		return -1;
	}
	/* checkCUDAError("compress"); */
    cuCtxSynchronize();
    kernel_time += probe_time_end(&ts_kernel);

	/*******************************************************
	 * COPY RESULTS BACK TO CPU
	 ******************************************************/
    probe_time_start(&ts_d2h);

	res = cuMemcpyDtoH(image, d_I, mem_size);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemcpyDtoH failed: res = %u\n", res);
		return -1;
	}
	/* checkCUDAError("copy back"); */
	//cuCtxSynchronize();
    d2h_time += probe_time_end(&ts_d2h);

	/*******************************************************
	 * CLEAN UP GPU
	 ******************************************************/

    probe_time_start(&ts_close);

	cuMemFree(d_I);
	cuMemFree(d_c);
	cuMemFree(d_iN);
	cuMemFree(d_iS);
	cuMemFree(d_jE);
	cuMemFree(d_jW);
	cuMemFree(d_dN);
	cuMemFree(d_dS);
	cuMemFree(d_dE);
	cuMemFree(d_dW);
	cuMemFree(d_sums);
	cuMemFree(d_sums2);

	res = cuda_driver_api_exit(ctx, mod);
	if (res != CUDA_SUCCESS) {
		pr_err("cuda_driver_api_exit faild: res = %u\n", res);
		return -1;
	}

    close_time = probe_time_end(&ts_close);
	total_time = probe_time_end(&ts_total);

    pr_info("Init: %ld\n", init_time);
	pr_info("MemAlloc: %ld\n", mem_alloc_time);
	pr_info("HtoD: %ld\n", h2d_time);
	pr_info("Exec: %ld\n", kernel_time);
	pr_info("DtoH: %ld\n", d2h_time);
	pr_info("Close: %ld\n", close_time);
	pr_info("API: %ld\n", init_time+mem_alloc_time+h2d_time+kernel_time+d2h_time+close_time);
	pr_info("Total: %ld (ns)\n", total_time);

	/*******************************************************
	 * WRITE IMAGE AFTER PROCESSING
	 ******************************************************/

    if (output_f)
    	write_graphics(output_f, image, Nr, Nc, 1, 255);

	/*******************************************************
	 * DEALLOCATE
	 ******************************************************/
	vfree(image_ori);
	vfree(image);
	vfree(iN);
	vfree(iS);
	vfree(jW);
	vfree(jE);

	return 0;
}

static void __exit drv_fini(void)
{
    pr_info("unload cuda srad_v1 module\n");
}

module_init(drv_init);
module_exit(drv_fini);

MODULE_AUTHOR("Hangchen Yu");
MODULE_DESCRIPTION("Driver module for Rodinia srad_v1");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
