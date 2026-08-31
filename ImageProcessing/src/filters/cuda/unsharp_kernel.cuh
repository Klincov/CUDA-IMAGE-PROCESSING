#pragma once
#include "core/CUDATiming.cuh"

void launchUnsharpMaskingKernel(const unsigned char* h_input, unsigned char* h_output, int width, int height, int channels,float amount);

CudaTimingResult launchUnsharpMaskingKernelTimed(const unsigned char* h_input, unsigned char* h_output, int width, int height, int channels,float amount);
