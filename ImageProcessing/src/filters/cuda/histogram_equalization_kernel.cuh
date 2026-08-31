#pragma once
#include "core/CUDATiming.cuh"

void launchHistogramEqualizationKernel(const unsigned char* h_input, unsigned char* h_output, int width, int height);

CudaTimingResult launchHistogramEqualizationKernelTimed(const unsigned char* h_input, unsigned char* h_output, int width, int height);
