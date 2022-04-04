#
# The following must be provided:
#

# A path to CUDA top directory
CUDA_TOP_DIR	:= ..
# An executable name
EXECUTABLE	:= needle
# Cuda source files (compiled with nvcc)
CUFILES		:= needle_cuda_kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= needle_cuda.c

# init C flags (more options will be added)
CFLAGS		:= -lm

include ../util/nvidia.mk
