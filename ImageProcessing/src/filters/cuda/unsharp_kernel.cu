#include "unsharp_kernel.cuh"
#include "gauss_kernel_internal.cuh"
#include "core/CudaCheck.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "core/CUDATiming.cuh"

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

    CUDA_CHECK(cudaMalloc(&d_input, totalBytes));
    CUDA_CHECK(cudaMalloc(&d_blurred, totalBytes));   // medjurezultat
    CUDA_CHECK(cudaMalloc(&d_output, totalBytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, totalBytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    gaussKernel << <blocks, threadsPerBlock >> > (d_input, d_blurred, width, height, channels);
    CUDA_CHECK_LAST_ERROR();

    unsharpKernel << <blocks, threadsPerBlock >> > (d_input, d_blurred, d_output, width, height, channels, amount);
    CUDA_CHECK_LAST_ERROR();

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, totalBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_output));
}

CudaTimingResult launchUnsharpMaskingKernelTimed(const unsigned char* h_input, unsigned char* h_output,
    int width, int height, int channels, float amount = 2) {

    int totalBytes = width * height * channels;

    unsigned char* d_input = nullptr;
    unsigned char* d_blurred = nullptr;
    unsigned char* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, totalBytes));
    CUDA_CHECK(cudaMalloc(&d_blurred, totalBytes));   // medjurezultat
    CUDA_CHECK(cudaMalloc(&d_output, totalBytes));

    cudaEvent_t startTotal, stopTotal;
    cudaEvent_t startH2D, stopH2D;
    cudaEvent_t startKernel, stopKernel; // obuhvata OBA kernela (gauss + unsharp)
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
    CUDA_CHECK(cudaMemcpy(d_input, h_input, totalBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stopH2D));

    // --- Kernel faza: gauss + unsharp ---
    dim3 threadsPerBlock(16, 16);
    dim3 blocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    CUDA_CHECK(cudaEventRecord(startKernel));

    gaussKernel << <blocks, threadsPerBlock >> > (d_input, d_blurred, width, height, channels);
    CUDA_CHECK_LAST_ERROR();

    unsharpKernel << <blocks, threadsPerBlock >> > (d_input, d_blurred, d_output, width, height, channels, amount);
    CUDA_CHECK_LAST_ERROR();

    CUDA_CHECK(cudaEventRecord(stopKernel));

    // --- D2H transfer ---
    CUDA_CHECK(cudaEventRecord(startD2H));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, totalBytes, cudaMemcpyDeviceToHost));
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

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}