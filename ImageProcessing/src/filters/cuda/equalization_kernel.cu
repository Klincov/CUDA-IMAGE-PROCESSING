#include "equalization_kernel.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define GRAY_LEVELS 256

__global__ void histogramKernel(const unsigned char* input, int* histogram,
    int width, int height) {

    __shared__ int localHist[GRAY_LEVELS];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int i = tid; i < GRAY_LEVELS; i += blockDim.x * blockDim.y)
        localHist[i] = 0;
    __syncthreads();

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        unsigned char pixel = input[row * width + col];
        atomicAdd(&localHist[pixel], 1);   // shared memory atomic
    }
    __syncthreads();

    // svaki blok sabira svoj lokalni histogram u globalni
    for (int i = tid; i < GRAY_LEVELS; i += blockDim.x * blockDim.y)
        atomicAdd(&histogram[i], localHist[i]);  // global memory atomic 
}

__global__ void equalizeMapKernel(const unsigned char* input, unsigned char* output,
    const int* cumSum, int cumSumMin, int totalPixels, int width, int height) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int idx = row * width + col;
    unsigned char pixel = input[idx];

    double equalized = round(
        (1.0 * cumSum[pixel] - cumSumMin) / (totalPixels - cumSumMin) * (GRAY_LEVELS - 1)
    );

    output[idx] = static_cast<unsigned char>(equalized);
}

void launchHistogramEqualizationKernel(const unsigned char* h_input, unsigned char* h_output,
    int width, int height) {

    int totalPixels = width * height;

    unsigned char* d_input, * d_output;
    int* d_histogram, * d_cumSum;

    cudaMalloc(&d_input, totalPixels);
    cudaMalloc(&d_output, totalPixels);
    cudaMalloc(&d_histogram, GRAY_LEVELS * sizeof(int));
    cudaMalloc(&d_cumSum, GRAY_LEVELS * sizeof(int));
    cudaMemset(d_histogram, 0, GRAY_LEVELS * sizeof(int));

    cudaMemcpy(d_input, h_input, totalPixels, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    // Faza 1: histogram (GPU)
    histogramKernel << <blocks, threadsPerBlock >> > (d_input, d_histogram, width, height);

    // Faza 2: cumsum preko cpu jer je 256 elemenata premalo za GPU
    int h_hist[GRAY_LEVELS];
    cudaMemcpy(h_hist, d_histogram, GRAY_LEVELS * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 1; i < GRAY_LEVELS; i++)
        h_hist[i] += h_hist[i - 1];

    int cumSumMin = 0;
    for (int i = 0; i < GRAY_LEVELS; i++) {
        if (h_hist[i] > 0) { cumSumMin = h_hist[i]; break; }
    }

    cudaMemcpy(d_cumSum, h_hist, GRAY_LEVELS * sizeof(int), cudaMemcpyHostToDevice);

    // Faza 3: mapiranje (GPU)
    equalizeMapKernel << <blocks, threadsPerBlock >> > (d_input, d_output, d_cumSum, cumSumMin, totalPixels, width, height);

    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, totalPixels, cudaMemcpyDeviceToHost);

    cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_histogram); cudaFree(d_cumSum);
}
