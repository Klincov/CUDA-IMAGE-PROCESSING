#include "invert_kernel.cuh"
#include <cuda_runtime.h>
#include "core/CudaCheck.cuh"
#include "core/CUDATiming.cuh"

__global__ void invertKernel(unsigned char* data, int totalBytes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalBytes) {
        data[idx] = 255 - data[idx];
    }
}

void launchInvertKernel(unsigned char* h_data, int width, int height, int channels) {
    int totalBytes = width * height * channels;
    unsigned char* d_data = nullptr;

    CUDA_CHECK(cudaMalloc(&d_data, totalBytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, totalBytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocks = (totalBytes + threadsPerBlock - 1) / threadsPerBlock;
    invertKernel << <blocks, threadsPerBlock >> > (d_data, totalBytes);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_data, d_data, totalBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
}

CudaTimingResult launchInvertKernelTimed(unsigned char* h_data, int width, int height, int channels) {
    int totalBytes = width * height * channels;
    unsigned char* d_data = nullptr;


    CUDA_CHECK(cudaMalloc(&d_data, totalBytes));

    cudaEvent_t startTotal, stopTotal;
    cudaEvent_t startH2D, stopH2D;
    cudaEvent_t startKernel, stopKernel;
    cudaEvent_t startD2H, stopD2H;

    CUDA_CHECK(cudaEventCreate(&startTotal));
    CUDA_CHECK(cudaEventCreate(&stopTotal));
    CUDA_CHECK(cudaEventCreate(&startH2D));
    CUDA_CHECK(cudaEventCreate(&stopH2D));
    CUDA_CHECK(cudaEventCreate(&startKernel));
    CUDA_CHECK(cudaEventCreate(&stopKernel));
    CUDA_CHECK(cudaEventCreate(&startD2H));
    CUDA_CHECK(cudaEventCreate(&stopD2H));

    CUDA_CHECK(cudaEventRecord(startTotal));

    // --- H2D transfer ---
    CUDA_CHECK(cudaEventRecord(startH2D));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, totalBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stopH2D));


    int threadsPerBlock = 256;
    int blocks = (totalBytes + threadsPerBlock - 1) / threadsPerBlock;

    // --- Kernel ---
    CUDA_CHECK(cudaEventRecord(startKernel));

    invertKernel << <blocks, threadsPerBlock >> > (d_data, totalBytes);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stopKernel));

    // --- D2H transfer --- 

    CUDA_CHECK(cudaEventRecord(startD2H));

    CUDA_CHECK(cudaMemcpy(h_data, d_data, totalBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stopD2H));
    CUDA_CHECK(cudaEventRecord(stopTotal));
    CUDA_CHECK(cudaEventSynchronize(stopTotal));

    CudaTimingResult result;
    CUDA_CHECK(cudaEventElapsedTime(&result.h2dMs, startH2D, stopH2D));
    CUDA_CHECK(cudaEventElapsedTime(&result.kernelMs, startKernel, stopKernel));
    CUDA_CHECK(cudaEventElapsedTime(&result.d2hMs, startD2H, stopD2H));
    CUDA_CHECK(cudaEventElapsedTime(&result.totalMs, startTotal, stopTotal));

    CUDA_CHECK(cudaEventDestroy(startTotal));
    CUDA_CHECK(cudaEventDestroy(stopTotal));
    CUDA_CHECK(cudaEventDestroy(startH2D));
    CUDA_CHECK(cudaEventDestroy(stopH2D));
    CUDA_CHECK(cudaEventDestroy(startKernel));
    CUDA_CHECK(cudaEventDestroy(stopKernel));
    CUDA_CHECK(cudaEventDestroy(startD2H));
    CUDA_CHECK(cudaEventDestroy(stopD2H));

    CUDA_CHECK(cudaFree(d_data));

    return result;
}