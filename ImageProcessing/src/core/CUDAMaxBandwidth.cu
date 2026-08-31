#pragma once
#include "CudaMaxBandwidth.cuh"
#include "core/CudaCheck.cuh"
#include <cuda_runtime.h>
#include <iostream>

void DisplayPeakBandwidth() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";

    int memoryClockRate = 0;
    int memoryBusWidth = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&memoryBusWidth, cudaDevAttrGlobalMemoryBusWidth, 0));

    double peakBandwidthGBps = 2.0 * memoryClockRate * (memoryBusWidth / 8.0) / 1.0e6;
    std::cout << "Peak Bandwidth (GBps): " << peakBandwidthGBps << "\n";

}