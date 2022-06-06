/**
 * Ported to kernel by Hangchen Yu.
 */
#define pr_fmt(fmt) "%s:%d:: " fmt, __func__, __LINE__
#include <linux/module.h>
#include <linux/delay.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>

#include "cuda_kava.h"

#include "../util/util.h"
#include "heartwall.h"
#include "AVI/avimod.h"

struct timestamp ts_init, ts_total, ts_memalloc, ts_h2d, ts_d2h, ts_kernel, ts_close;
long init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0, d2h_phy_time = 0, h2d_phy_time = 0;

static int nframes = 0;
static char *input_f = NULL;
static char *cubin_p = NULL;
module_param(nframes, int, S_IRUSR);
MODULE_PARM_DESC(nframes, "Number of frames");
module_param(input_f, charp, 0000);
MODULE_PARM_DESC(input_f, "Input video path");
module_param(cubin_p, charp, 0000);
MODULE_PARM_DESC(cubin_p, "CUDA binary path");

//=========================================================================
//	STRUCTURES, GLOBAL STRUCTURE VARIABLES
//=========================================================================

params_common_change common_change;

params_common common;

// cannot determine size dynamically so choose more than usually needed
params_unique unique[ALL_POINTS];

CUdeviceptr d_common_change;
size_t d_common_change_size;

CUdeviceptr d_common;
size_t d_common_size;

CUdeviceptr d_unique;
size_t d_unique_size;

//=========================================================================
// KERNEL CODE
//=========================================================================

CUresult heartwall_launch(CUmodule mod, int gdx, int gdy, int bdx, int bdy)
{
	void* param[] = {}; /* no params */
	CUfunction f;
	CUresult res;

	res = cuModuleGetFunction(&f, mod, "_Z6kernelv");
	if (res != CUDA_SUCCESS) {
		pr_err("cuModuleGetFunction failed: res = %u\n", res);
		return res;
	}

	/* shared memory size is known in the kernel image. */
	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**) param, NULL);
	if (res != CUDA_SUCCESS) {
		pr_err("cuLaunchKernel(euclid) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

//=========================================================================
//=========================================================================
//	MAIN FUNCTION
//=========================================================================
//=========================================================================

static int __init drv_init(void)
{
	//=====================================================================
	//	VARIABLES
	//=====================================================================
    char cubin_fn[128];

	// CUDA kernel execution parameters
	int gdx, gdy, bdx, bdy;

	CUcontext ctx;
	CUmodule mod;
	CUresult res;

	// counter
	int i;
	int frames_processed;

	// frames
	char* video_file_name;
	avi_t* frames;
	int* frame;

    pr_info("load cuda heartwall module\n");

	//=====================================================================
	// 	FRAME
	//=====================================================================

	if (!input_f || !cubin_p || nframes == 0){
		pr_err("Usage: insmod heartwall.ko input_f=<inputfile> nframes=<num of frames> cubin_p=<path to cubin>\n");
		return -1;
	}
	
	// open movie file
 	video_file_name = input_f;
	frames = (avi_t*)AVI_open_input_file(video_file_name, 1);		// added casting
	if (frames == NULL)  {
        AVI_print_error((char *) "Error with AVI_open_input_file");
        return -1;
	}

	// common
	common.no_frames = AVI_video_frames(frames);
	common.frame_rows = AVI_video_height(frames);
	common.frame_cols = AVI_video_width(frames);
	common.frame_elem = common.frame_rows * common.frame_cols;
	common.frame_mem = sizeof(fp) * common.frame_elem;

	//=====================================================================
	// 	CHECK INPUT ARGUMENTS
	//=====================================================================

	frames_processed = nframes;
    if(frames_processed<0 || frames_processed>common.no_frames){
        pr_err("ERROR: %d is an incorrect number of frames specified, select in the range of 0-%d\n", frames_processed, common.no_frames);
        return 0;
	}

	//=====================================================================
	// DRIVER INIT
	//=====================================================================
    strcpy(cubin_fn, cubin_p);
    strcat(cubin_fn, "/heartwall.cubin");

    probe_time_start(&ts_total);
    probe_time_start(&ts_init);

	res = cuda_driver_api_init(&ctx, &mod, cubin_fn);
	if (res != CUDA_SUCCESS) {
		pr_err("cuda_driver_api_init failed: res = %u\n", res);
		return -1;
	}

    init_time = probe_time_end(&ts_init);

    probe_time_start(&ts_memalloc);

	// pointers
	cuMemAlloc((CUdeviceptr*)&common_change.d_frame, common.frame_mem);

	//=====================================================================
	//	HARDCODED INPUTS FROM MATLAB
	//=====================================================================

	//=====================================================================
	//	CONSTANTS
	//=====================================================================

	common.sSize = 40;
	common.tSize = 25;
	common.maxMove = 10;
	common.alpha = 0.87;

	//=====================================================================
	//	ENDO POINTS
	//=====================================================================

	common.endoPoints = ENDO_POINTS;
	common.endo_mem = sizeof(int) * common.endoPoints;

	common.endoRow = (int *)vmalloc(common.endo_mem);
	common.endoRow[ 0] = 369;
	common.endoRow[ 1] = 400;
	common.endoRow[ 2] = 429;
	common.endoRow[ 3] = 452;
	common.endoRow[ 4] = 476;
	common.endoRow[ 5] = 486;
	common.endoRow[ 6] = 479;
	common.endoRow[ 7] = 458;
	common.endoRow[ 8] = 433;
	common.endoRow[ 9] = 404;
	common.endoRow[10] = 374;
	common.endoRow[11] = 346;
	common.endoRow[12] = 318;
	common.endoRow[13] = 294;
	common.endoRow[14] = 277;
	common.endoRow[15] = 269;
	common.endoRow[16] = 275;
	common.endoRow[17] = 287;
	common.endoRow[18] = 311;
	common.endoRow[19] = 339;
	cuMemAlloc((CUdeviceptr*)&common.d_endoRow, common.endo_mem);

	common.endoCol = (int *)vmalloc(common.endo_mem);
	common.endoCol[ 0] = 408;
	common.endoCol[ 1] = 406;
	common.endoCol[ 2] = 397;
	common.endoCol[ 3] = 383;
	common.endoCol[ 4] = 354;
	common.endoCol[ 5] = 322;
	common.endoCol[ 6] = 294;
	common.endoCol[ 7] = 270;
	common.endoCol[ 8] = 250;
	common.endoCol[ 9] = 237;
	common.endoCol[10] = 235;
	common.endoCol[11] = 241;
	common.endoCol[12] = 254;
	common.endoCol[13] = 273;
	common.endoCol[14] = 300;
	common.endoCol[15] = 328;
	common.endoCol[16] = 356;
	common.endoCol[17] = 383;
	common.endoCol[18] = 401;
	common.endoCol[19] = 411;
	cuMemAlloc((CUdeviceptr*)&common.d_endoCol, common.endo_mem);

	common.tEndoRowLoc = (int *)vmalloc(common.endo_mem * common.no_frames);
	cuMemAlloc((CUdeviceptr*)&common.d_tEndoRowLoc, common.endo_mem * common.no_frames);

	common.tEndoColLoc = (int *)vmalloc(common.endo_mem * common.no_frames);
	cuMemAlloc((CUdeviceptr*)&common.d_tEndoColLoc, common.endo_mem * common.no_frames);

	//=====================================================================
	//	EPI POINTS
	//=====================================================================

	common.epiPoints = EPI_POINTS;
	common.epi_mem = sizeof(int) * common.epiPoints;

	common.epiRow = (int *)vmalloc(common.epi_mem);
	common.epiRow[ 0] = 390;
	common.epiRow[ 1] = 419;
	common.epiRow[ 2] = 448;
	common.epiRow[ 3] = 474;
	common.epiRow[ 4] = 501;
	common.epiRow[ 5] = 519;
	common.epiRow[ 6] = 535;
	common.epiRow[ 7] = 542;
	common.epiRow[ 8] = 543;
	common.epiRow[ 9] = 538;
	common.epiRow[10] = 528;
	common.epiRow[11] = 511;
	common.epiRow[12] = 491;
	common.epiRow[13] = 466;
	common.epiRow[14] = 438;
	common.epiRow[15] = 406;
	common.epiRow[16] = 376;
	common.epiRow[17] = 347;
	common.epiRow[18] = 318;
	common.epiRow[19] = 291;
	common.epiRow[20] = 275;
	common.epiRow[21] = 259;
	common.epiRow[22] = 256;
	common.epiRow[23] = 252;
	common.epiRow[24] = 252;
	common.epiRow[25] = 257;
	common.epiRow[26] = 266;
	common.epiRow[27] = 283;
	common.epiRow[28] = 305;
	common.epiRow[29] = 331;
	common.epiRow[30] = 360;
	cuMemAlloc((CUdeviceptr*)&common.d_epiRow, common.epi_mem);

	common.epiCol = (int *)vmalloc(common.epi_mem);
	common.epiCol[ 0] = 457;
	common.epiCol[ 1] = 454;
	common.epiCol[ 2] = 446;
	common.epiCol[ 3] = 431;
	common.epiCol[ 4] = 411;
	common.epiCol[ 5] = 388;
	common.epiCol[ 6] = 361;
	common.epiCol[ 7] = 331;
	common.epiCol[ 8] = 301;
	common.epiCol[ 9] = 273;
	common.epiCol[10] = 243;
	common.epiCol[11] = 218;
	common.epiCol[12] = 196;
	common.epiCol[13] = 178;
	common.epiCol[14] = 166;
	common.epiCol[15] = 157;
	common.epiCol[16] = 155;
	common.epiCol[17] = 165;
	common.epiCol[18] = 177;
	common.epiCol[19] = 197;
	common.epiCol[20] = 218;
	common.epiCol[21] = 248;
	common.epiCol[22] = 276;
	common.epiCol[23] = 304;
	common.epiCol[24] = 333;
	common.epiCol[25] = 361;
	common.epiCol[26] = 391;
	common.epiCol[27] = 415;
	common.epiCol[28] = 434;
	common.epiCol[29] = 448;
	common.epiCol[30] = 455;
	cuMemAlloc((CUdeviceptr*)&common.d_epiCol, common.epi_mem);

	common.tEpiRowLoc = (int *)vmalloc(common.epi_mem * common.no_frames);
	cuMemAlloc((CUdeviceptr*)&common.d_tEpiRowLoc, common.epi_mem * common.no_frames);

	common.tEpiColLoc = (int *)vmalloc(common.epi_mem * common.no_frames);
	cuMemAlloc((CUdeviceptr*)&common.d_tEpiColLoc, common.epi_mem * common.no_frames);

	//=====================================================================
	//	ALL POINTS
	//=====================================================================

	common.allPoints = ALL_POINTS;

	//=====================================================================
	// 	TEMPLATE SIZES
	//=====================================================================

	// common
	common.in_rows = common.tSize + 1 + common.tSize;
	common.in_cols = common.in_rows;
	common.in_elem = common.in_rows * common.in_cols;
	common.in_mem = sizeof(fp) * common.in_elem;

	//=====================================================================
	// 	CREATE ARRAY OF TEMPLATES FOR ALL POINTS
	//=====================================================================

	// common
	cuMemAlloc((CUdeviceptr*)&common.d_endoT, common.in_mem * common.endoPoints);
	cuMemAlloc((CUdeviceptr*)&common.d_epiT, common.in_mem * common.epiPoints);

	//=====================================================================
	//	SPECIFIC TO ENDO OR EPI TO BE SET HERE
	//=====================================================================

	for(i=0; i<common.endoPoints; i++){
		unique[i].point_no = i;
		unique[i].d_Row = common.d_endoRow;
		unique[i].d_Col = common.d_endoCol;
		unique[i].d_tRowLoc = common.d_tEndoRowLoc;
		unique[i].d_tColLoc = common.d_tEndoColLoc;
		unique[i].d_T = common.d_endoT;
	}
	for(i=common.endoPoints; i<common.allPoints; i++){
		unique[i].point_no = i-common.endoPoints;
		unique[i].d_Row = common.d_epiRow;
		unique[i].d_Col = common.d_epiCol;
		unique[i].d_tRowLoc = common.d_tEpiRowLoc;
		unique[i].d_tColLoc = common.d_tEpiColLoc;
		unique[i].d_T = common.d_epiT;
	}

	//=====================================================================
	// 	RIGHT TEMPLATE 	FROM 	TEMPLATE ARRAY
	//=====================================================================

	// pointers
	for(i=0; i<common.allPoints; i++){
		unique[i].in_pointer = unique[i].point_no * common.in_elem;
	}

	//=====================================================================
	// 	AREA AROUND POINT		FROM	FRAME
	//=====================================================================

	// common
	common.in2_rows = 2 * common.sSize + 1;
	common.in2_cols = 2 * common.sSize + 1;
	common.in2_elem = common.in2_rows * common.in2_cols;
	common.in2_mem = sizeof(float) * common.in2_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cuMemAlloc((CUdeviceptr*)&unique[i].d_in2, common.in2_mem);
	}

	//=====================================================================
	// 	CONVOLUTION
	//=====================================================================

	// common
	common.conv_rows = common.in_rows + common.in2_rows - 1;												// number of rows in I
	common.conv_cols = common.in_cols + common.in2_cols - 1;												// number of columns in I
	common.conv_elem = common.conv_rows * common.conv_cols;												// number of elements
	common.conv_mem = sizeof(float) * common.conv_elem;
	common.ioffset = 0;
	common.joffset = 0;

	// pointers
	for(i=0; i<common.allPoints; i++){
		res = cuMemAlloc((CUdeviceptr*)&unique[i].d_conv, common.conv_mem);
	}

	//=====================================================================
	// 	CUMULATIVE SUM
	//=====================================================================

	//=====================================================================
	// 	PADDING OF ARRAY, VERTICAL CUMULATIVE SUM
	//=====================================================================

	// common
	common.in2_pad_add_rows = common.in_rows;
	common.in2_pad_add_cols = common.in_cols;

	common.in2_pad_cumv_rows = common.in2_rows + 2*common.in2_pad_add_rows;
	common.in2_pad_cumv_cols = common.in2_cols + 2*common.in2_pad_add_cols;
	common.in2_pad_cumv_elem = common.in2_pad_cumv_rows * common.in2_pad_cumv_cols;
	common.in2_pad_cumv_mem = sizeof(float) * common.in2_pad_cumv_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cuMemAlloc((CUdeviceptr*)&unique[i].d_in2_pad_cumv, common.in2_pad_cumv_mem);
	}

	//=====================================================================
	// 	SELECTION
	//=====================================================================

	// common
	common.in2_pad_cumv_sel_rowlow = 1 + common.in_rows;													// (1 to n+1)
	common.in2_pad_cumv_sel_rowhig = common.in2_pad_cumv_rows - 1;
	common.in2_pad_cumv_sel_collow = 1;
	common.in2_pad_cumv_sel_colhig = common.in2_pad_cumv_cols;
	common.in2_pad_cumv_sel_rows = common.in2_pad_cumv_sel_rowhig - common.in2_pad_cumv_sel_rowlow + 1;
	common.in2_pad_cumv_sel_cols = common.in2_pad_cumv_sel_colhig - common.in2_pad_cumv_sel_collow + 1;
	common.in2_pad_cumv_sel_elem = common.in2_pad_cumv_sel_rows * common.in2_pad_cumv_sel_cols;
	common.in2_pad_cumv_sel_mem = sizeof(float) * common.in2_pad_cumv_sel_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cuMemAlloc((CUdeviceptr*)&unique[i].d_in2_pad_cumv_sel, common.in2_pad_cumv_sel_mem);
	}

	//=====================================================================
	// 	SELECTION	2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
	//=====================================================================

	// common
	common.in2_pad_cumv_sel2_rowlow = 1;
	common.in2_pad_cumv_sel2_rowhig = common.in2_pad_cumv_rows - common.in_rows - 1;
	common.in2_pad_cumv_sel2_collow = 1;
	common.in2_pad_cumv_sel2_colhig = common.in2_pad_cumv_cols;
	common.in2_sub_cumh_rows = common.in2_pad_cumv_sel2_rowhig - common.in2_pad_cumv_sel2_rowlow + 1;
	common.in2_sub_cumh_cols = common.in2_pad_cumv_sel2_colhig - common.in2_pad_cumv_sel2_collow + 1;
	common.in2_sub_cumh_elem = common.in2_sub_cumh_rows * common.in2_sub_cumh_cols;
	common.in2_sub_cumh_mem = sizeof(float) * common.in2_sub_cumh_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cuMemAlloc((CUdeviceptr*)&unique[i].d_in2_sub_cumh, common.in2_sub_cumh_mem);
	}

	//=====================================================================
	// 	SELECTION
	//=====================================================================

	// common
	common.in2_sub_cumh_sel_rowlow = 1;
	common.in2_sub_cumh_sel_rowhig = common.in2_sub_cumh_rows;
	common.in2_sub_cumh_sel_collow = 1 + common.in_cols;
	common.in2_sub_cumh_sel_colhig = common.in2_sub_cumh_cols - 1;
	common.in2_sub_cumh_sel_rows = common.in2_sub_cumh_sel_rowhig - common.in2_sub_cumh_sel_rowlow + 1;
	common.in2_sub_cumh_sel_cols = common.in2_sub_cumh_sel_colhig - common.in2_sub_cumh_sel_collow + 1;
	common.in2_sub_cumh_sel_elem = common.in2_sub_cumh_sel_rows * common.in2_sub_cumh_sel_cols;
	common.in2_sub_cumh_sel_mem = sizeof(float) * common.in2_sub_cumh_sel_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cuMemAlloc((CUdeviceptr*)&unique[i].d_in2_sub_cumh_sel, common.in2_sub_cumh_sel_mem);
	}

	//=====================================================================
	//	SELECTION 2, SUBTRACTION
	//=====================================================================

	// common
	common.in2_sub_cumh_sel2_rowlow = 1;
	common.in2_sub_cumh_sel2_rowhig = common.in2_sub_cumh_rows;
	common.in2_sub_cumh_sel2_collow = 1;
	common.in2_sub_cumh_sel2_colhig = common.in2_sub_cumh_cols - common.in_cols - 1;
	common.in2_sub2_rows = common.in2_sub_cumh_sel2_rowhig - common.in2_sub_cumh_sel2_rowlow + 1;
	common.in2_sub2_cols = common.in2_sub_cumh_sel2_colhig - common.in2_sub_cumh_sel2_collow + 1;
	common.in2_sub2_elem = common.in2_sub2_rows * common.in2_sub2_cols;
	common.in2_sub2_mem = sizeof(float) * common.in2_sub2_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cuMemAlloc((CUdeviceptr*)&unique[i].d_in2_sub2, common.in2_sub2_mem);
	}

	//=====================================================================
	//	CUMULATIVE SUM 2
	//=====================================================================

	//=====================================================================
	//	MULTIPLICATION
	//=====================================================================

	// common
	common.in2_sqr_rows = common.in2_rows;
	common.in2_sqr_cols = common.in2_cols;
	common.in2_sqr_elem = common.in2_elem;
	common.in2_sqr_mem = common.in2_mem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cuMemAlloc((CUdeviceptr*)&unique[i].d_in2_sqr, common.in2_sqr_mem);
	}

	//=====================================================================
	//	SELECTION 2, SUBTRACTION
	//=====================================================================

	// common
	common.in2_sqr_sub2_rows = common.in2_sub2_rows;
	common.in2_sqr_sub2_cols = common.in2_sub2_cols;
	common.in2_sqr_sub2_elem = common.in2_sub2_elem;
	common.in2_sqr_sub2_mem = common.in2_sub2_mem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cuMemAlloc((CUdeviceptr*)&unique[i].d_in2_sqr_sub2, common.in2_sqr_sub2_mem);
	}

	//=====================================================================
	//	FINAL
	//=====================================================================

	// common
	common.in_sqr_rows = common.in_rows;
	common.in_sqr_cols = common.in_cols;
	common.in_sqr_elem = common.in_elem;
	common.in_sqr_mem = common.in_mem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cuMemAlloc((CUdeviceptr*)&unique[i].d_in_sqr, common.in_sqr_mem);
	}

	//=====================================================================
	//	TEMPLATE MASK CREATE
	//=====================================================================

	// common
	common.tMask_rows = common.in_rows + (common.sSize+1+common.sSize) - 1;
	common.tMask_cols = common.tMask_rows;
	common.tMask_elem = common.tMask_rows * common.tMask_cols;
	common.tMask_mem = sizeof(float) * common.tMask_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cuMemAlloc((CUdeviceptr*)&unique[i].d_tMask, common.tMask_mem);
	}

	//=====================================================================
	//	POINT MASK INITIALIZE
	//=====================================================================

	// common
	common.mask_rows = common.maxMove;
	common.mask_cols = common.mask_rows;
	common.mask_elem = common.mask_rows * common.mask_cols;
	common.mask_mem = sizeof(float) * common.mask_elem;

	//=====================================================================
	//	MASK CONVOLUTION
	//=====================================================================

	// common
	common.mask_conv_rows = common.tMask_rows;												// number of rows in I
	common.mask_conv_cols = common.tMask_cols;												// number of columns in I
	common.mask_conv_elem = common.mask_conv_rows * common.mask_conv_cols;												// number of elements
	common.mask_conv_mem = sizeof(float) * common.mask_conv_elem;
	common.mask_conv_ioffset = (common.mask_rows-1)/2;
	if((common.mask_rows-1) % 2 > 0) {
		common.mask_conv_ioffset = common.mask_conv_ioffset + 1;
	}
	common.mask_conv_joffset = (common.mask_cols-1)/2;
	if((common.mask_cols-1) % 2 > 0){
		common.mask_conv_joffset = common.mask_conv_joffset + 1;
	}

	// pointers
	for(i=0; i<common.allPoints; i++){
		cuMemAlloc((CUdeviceptr*)&unique[i].d_mask_conv, common.mask_conv_mem);
	}

    mem_alloc_time = probe_time_end(&ts_memalloc);
    probe_time_start(&ts_h2d);

	//=====================================================================
	//	COPY
	//=====================================================================

	cuMemcpyHtoD((CUdeviceptr)common.d_endoRow, common.endoRow, common.endo_mem);
	cuMemcpyHtoD((CUdeviceptr)common.d_endoCol, common.endoCol, common.endo_mem);
	cuMemcpyHtoD((CUdeviceptr)common.d_epiRow, common.epiRow, common.epi_mem);
	cuMemcpyHtoD((CUdeviceptr)common.d_epiCol, common.epiCol, common.epi_mem);

	//=====================================================================
	//	COPY ARGUMENTS
	//=====================================================================

	cuModuleGetGlobal(&d_common, &d_common_size, mod, "d_common");
	cuMemcpyHtoD(d_common, &common, d_common_size);

	cuModuleGetGlobal(&d_unique, &d_unique_size, mod, "d_unique");
	cuMemcpyHtoD(d_unique, &unique, d_unique_size);

    h2d_time = probe_time_end(&ts_h2d);

	//=====================================================================
	//	KERNEL
	//=====================================================================

	//=====================================================================
	//	THREAD BLOCK
	//=====================================================================

	// All kernels operations within kernel use same max size of threads. Size of block size is set to the size appropriate for max size operation (on padded matrix). Other use subsets of that.
	bdx = NUMBER_THREADS; // define the number of threads in the block
	bdy = 1;
	gdx = common.allPoints;	// define the number of blocks in the grid
	gdy = 1;

	//=====================================================================
	//	PRINT FRAME PROGRESS START
	//=====================================================================

	pr_info("frame progress: ");

	//=====================================================================
	//	LAUNCH
	//=====================================================================

	for(common_change.frame_no=0; common_change.frame_no<frames_processed; common_change.frame_no++){

		// Extract a cropped version of the first frame from the video file
		frame = get_frame(frames,						// pointer to video file
						common_change.frame_no,				// number of frame that needs to be returned
						0,								// cropped?
						0,								// scaled?
						1);							// converted

		// copy frame to GPU memory
        probe_time_start(&ts_h2d);

		cuMemcpyHtoD((CUdeviceptr)common_change.d_frame, frame, common.frame_mem);
		cuModuleGetGlobal(&d_common_change, &d_common_change_size, mod, "d_common_change");
		cuMemcpyHtoD(d_common_change, &common_change, d_common_change_size);

        h2d_time += probe_time_end(&ts_h2d);

		// launch GPU kernel
        probe_time_start(&ts_kernel);

		res = heartwall_launch(mod, gdx, gdy, bdx, bdy);
		if (res != CUDA_SUCCESS) {
			pr_err("heartwall_launch failed: res = %u\n", res);
			return -1;
		}

        cuCtxSynchronize();
        kernel_time += probe_time_end(&ts_kernel);

		// free frame after each loop iteration, since AVI library allocates memory for every frame fetched
		vfree(frame);

		// print frame progress
		pr_info("%d ", common_change.frame_no);
	}

	//=====================================================================
	//	PRINT FRAME PROGRESS END
	//=====================================================================

	pr_info("\n");

	//=====================================================================
	//	OUTPUT
	//=====================================================================

    probe_time_start(&ts_d2h);

	cuMemcpyDtoH(common.tEndoRowLoc, (CUdeviceptr)common.d_tEndoRowLoc, common.endo_mem * common.no_frames);
	cuMemcpyDtoH(common.tEndoColLoc, (CUdeviceptr)common.d_tEndoColLoc, common.endo_mem * common.no_frames);

	cuMemcpyDtoH(common.tEpiRowLoc, (CUdeviceptr)common.d_tEpiRowLoc, common.epi_mem * common.no_frames);
	cuMemcpyDtoH(common.tEpiColLoc, (CUdeviceptr)common.d_tEpiColLoc, common.epi_mem * common.no_frames);

    d2h_time += probe_time_end(&ts_d2h);
    probe_time_start(&ts_close);

	//=====================================================================
	//	DEALLOCATION
	//=====================================================================

	//=====================================================================
	//	COMMON
	//=====================================================================

	// frame
	cuMemFree((CUdeviceptr)common_change.d_frame);

	// endo points
	vfree(common.endoRow);
	vfree(common.endoCol);
	vfree(common.tEndoRowLoc);
	vfree(common.tEndoColLoc);

	cuMemFree((CUdeviceptr)common.d_endoRow);
	cuMemFree((CUdeviceptr)common.d_endoCol);
	cuMemFree((CUdeviceptr)common.d_tEndoRowLoc);
	cuMemFree((CUdeviceptr)common.d_tEndoColLoc);

	cuMemFree((CUdeviceptr)common.d_endoT);

	// epi points
	vfree(common.epiRow);
	vfree(common.epiCol);
	vfree(common.tEpiRowLoc);
	vfree(common.tEpiColLoc);

	cuMemFree((CUdeviceptr)common.d_epiRow);
	cuMemFree((CUdeviceptr)common.d_epiCol);
	cuMemFree((CUdeviceptr)common.d_tEpiRowLoc);
	cuMemFree((CUdeviceptr)common.d_tEpiColLoc);

	cuMemFree((CUdeviceptr)common.d_epiT);

	//=====================================================================
	//	POINTERS
	//=====================================================================

	for(i=0; i<common.allPoints; i++){
		cuMemFree((CUdeviceptr)unique[i].d_in2);

		cuMemFree((CUdeviceptr)unique[i].d_conv);
		cuMemFree((CUdeviceptr)unique[i].d_in2_pad_cumv);
		cuMemFree((CUdeviceptr)unique[i].d_in2_pad_cumv_sel);
		cuMemFree((CUdeviceptr)unique[i].d_in2_sub_cumh);
		cuMemFree((CUdeviceptr)unique[i].d_in2_sub_cumh_sel);
		cuMemFree((CUdeviceptr)unique[i].d_in2_sub2);
		cuMemFree((CUdeviceptr)unique[i].d_in2_sqr);
		cuMemFree((CUdeviceptr)unique[i].d_in2_sqr_sub2);
		cuMemFree((CUdeviceptr)unique[i].d_in_sqr);

		cuMemFree((CUdeviceptr)unique[i].d_tMask);
		cuMemFree((CUdeviceptr)unique[i].d_mask_conv);
	}

	//=====================================================================
	// DRIVER EXIT
	//=====================================================================
	res = cuda_driver_api_exit(ctx, mod);
	if (res != CUDA_SUCCESS) {
		pr_err("cuda_driver_api_exit failed: res = %u\n", res);
		return -1;
	}

    close_time += probe_time_end(&ts_close);
	total_time = probe_time_end(&ts_total);

	pr_info("Init: %ld\n", init_time);
	pr_info("MemAlloc: %ld\n", mem_alloc_time);
	pr_info("HtoD: %ld\n", h2d_time);
	pr_info("Exec: %ld\n", kernel_time);
	pr_info("DtoH: %ld\n", d2h_time);
	pr_info("Close: %ld\n", close_time);
	pr_info("API: %ld\n", init_time+mem_alloc_time+h2d_time+kernel_time+d2h_time+close_time);
	pr_info("Total: %ld (ns)\n", total_time);

	return 0;
}

static void __exit drv_fini(void)
{
    pr_info("unload cuda heartwall module\n");
}

module_init(drv_init);
module_exit(drv_fini);

MODULE_AUTHOR("Hangchen Yu");
MODULE_DESCRIPTION("Driver module for Rodinia heartwall");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
