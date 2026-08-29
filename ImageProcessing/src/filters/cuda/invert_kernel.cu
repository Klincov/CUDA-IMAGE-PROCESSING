#include "invert_kernel.cuh"
#include <cuda_runtime.h>
#include "core/CudaCheck.cuh"

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