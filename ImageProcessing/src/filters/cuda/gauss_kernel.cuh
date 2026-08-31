#pragma once
#include "core/CudaTiming.cuh"

void launchGaussianBlurKernel(const unsigned char* h_input, unsigned char* h_output,int width, int height, int channels);

CudaTimingResult launchGaussianBlurKernelTimed(const unsigned char* h_input, unsigned char* h_output,
    int width, int height, int channels);
