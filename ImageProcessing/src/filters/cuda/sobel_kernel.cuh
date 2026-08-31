#pragma once
#include "core/CUDATiming.cuh"

void launchSobelFilterKernel(const unsigned char* h_input, unsigned char* h_output, int width, int height);

CudaTimingResult launchSobelFilterKernelTimed(const unsigned char* h_input, unsigned char* h_output, int width, int height);
