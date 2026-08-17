#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void gaussKernel(const unsigned char* input, unsigned char* output,
    int width, int height, int channels);

__device__ __forceinline__ int clampInt(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}