#pragma once
#include "core/CUDATiming.cuh"

void launchInvertKernel(unsigned char* h_data, int width, int height, int channels);

CudaTimingResult launchInvertKernelTimed(unsigned char* h_data, int width, int height, int channels);
