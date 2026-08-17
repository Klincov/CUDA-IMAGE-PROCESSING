#include "unsharp_kernel.cuh"
#include "gauss_kernel_internal.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void unsharpKernel(const unsigned char* original, const unsigned char* blurred,
    unsigned char* output, int width, int height, int channels, float amount) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;

    int idx = (row * width + col) * channels;

    for (int c = 0; c < channels; c++) {
        int detail = original[idx + c] - blurred[idx + c];
        int result = original[idx + c] + static_cast<int>(amount * detail);
        output[idx + c] = static_cast<unsigned char>(clampInt(result, 0, 255));
    }
}

void launchUnsharpMaskingKernel(const unsigned char* h_input, unsigned char* h_output,
    int width, int height, int channels, float amount = 2) {

    int totalBytes = width * height * channels;

    unsigned char* d_input = nullptr;
    unsigned char* d_blurred = nullptr;
    unsigned char* d_output = nullptr;

    cudaMalloc(&d_input, totalBytes);
    cudaMalloc(&d_blurred, totalBytes);   // medjurezultat
    cudaMalloc(&d_output, totalBytes);

    cudaMemcpy(d_input, h_input, totalBytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    gaussKernel << <blocks, threadsPerBlock >> > (d_input, d_blurred, width, height, channels);

    unsharpKernel << <blocks, threadsPerBlock >> > (d_input, d_blurred, d_output, width, height, channels, amount);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, totalBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_blurred);
    cudaFree(d_output);
}