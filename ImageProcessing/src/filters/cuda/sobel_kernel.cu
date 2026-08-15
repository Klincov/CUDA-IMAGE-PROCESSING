#include "sobel_kernel.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//constant memorija je kesirana i broadcast-uje se svim nitima u warpu
__constant__ int G_x[9] = { -1, 0, 1,
                            -2, 0, 2,
                            -1, 0, 1 };

__constant__ int G_y[9] = { -1, -2, -1,
                             0, 0, 0,
                             1, 2, 1 };




__device__ __forceinline__ int clampInt(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ int abs(int v) {
    if (v < 0) return -v;
    else return v;
}

__global__ void sobelKernel(const unsigned char* input, unsigned char* output,
    int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;

    int sum_x = 0;
    int sum_y = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int targetCol = clampInt(col + j, 0, width - 1);
            int targetRow = clampInt(row + i, 0, height - 1);
            int weight_x = G_x[(i + 1) * 3 + (j + 1)];
            int weight_y = G_y[(i + 1) * 3 + (j + 1)];


            int pixelIdx = targetRow * width + targetCol;
            sum_x += input[pixelIdx] * weight_x;
            sum_y += input[pixelIdx] * weight_y;


        }
    }

    int outIdx = row * width + col;
    int result = abs(sum_x) + abs(sum_y) >= 255 ? 255 : abs(sum_x) + abs(sum_y);

    output[outIdx] = static_cast<unsigned char>(result);
}

void launchSobelFilterKernel(const unsigned char* h_input, unsigned char* h_output,
    int width, int height) {
    int totalBytes = width * height;

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    cudaMalloc(&d_input, totalBytes);
    cudaMalloc(&d_output, totalBytes);

    cudaMemcpy(d_input, h_input, totalBytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    sobelKernel << <blocks, threadsPerBlock >> > (d_input, d_output, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, totalBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
