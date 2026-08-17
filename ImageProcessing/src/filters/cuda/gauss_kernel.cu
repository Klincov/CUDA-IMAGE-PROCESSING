#include "gauss_kernel.cuh"
#include "gauss_kernel_internal.cuh"  

//constant memorija je kesirana i broadcast-uje se svim nitima u warpu
__constant__ int d_kernel[25] = {
    1,  4,  7,  4, 1,
    4, 16, 26, 16, 4,
    7, 26, 41, 26, 7,
    4, 16, 26, 16, 4,
    1,  4,  7,  4, 1
};

__global__ void gaussKernel(const unsigned char* input, unsigned char* output,
    int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;

    int sums[4] = { 0, 0, 0, 0 }; // do 4 kanala (BGR ili BGRA)

    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            int targetCol = clampInt(col + j, 0, width - 1);
            int targetRow = clampInt(row + i, 0, height - 1);
            int weight = d_kernel[(i + 2) * 5 + (j + 2)];

            int pixelIdx = (targetRow * width + targetCol) * channels;
            for (int c = 0; c < channels; c++) {
                sums[c] += input[pixelIdx + c] * weight;
            }
        }
    }

    int outIdx = (row * width + col) * channels;
    for (int c = 0; c < channels; c++) {
        output[outIdx + c] = static_cast<unsigned char>(sums[c] / 273);
    }
}

void launchGaussianBlurKernel(const unsigned char* h_input, unsigned char* h_output,
    int width, int height, int channels) {
    int totalBytes = width * height * channels;

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

    gaussKernel << <blocks, threadsPerBlock >> > (d_input, d_output, width, height, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, totalBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
