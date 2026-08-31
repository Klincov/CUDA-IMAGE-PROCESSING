#include "gauss_kernel.cuh"
#include "gauss_kernel_internal.cuh"
#include "core/CudaCheck.cuh"
#include "core/CudaTiming.cuh"


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

    CUDA_CHECK(cudaMalloc(&d_input, totalBytes));
    CUDA_CHECK(cudaMalloc(&d_output, totalBytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, totalBytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    gaussKernel << <blocks, threadsPerBlock >> > (d_input, d_output, width, height, channels);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, totalBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}


CudaTimingResult launchGaussianBlurKernelTimed(const unsigned char* h_input, unsigned char* h_output,
    int width, int height, int channels) {
    int totalBytes = width * height * channels;

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, totalBytes));
    CUDA_CHECK(cudaMalloc(&d_output, totalBytes));

    // Event-ovi za merenje - jedan par po fazi, plus start/stop za total
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
    CUDA_CHECK(cudaMemcpy(d_input, h_input, totalBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stopH2D));

    // --- Kernel ---
    dim3 threadsPerBlock(16, 16);
    dim3 blocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    CUDA_CHECK(cudaEventRecord(startKernel));
    gaussKernel << <blocks, threadsPerBlock >> > (d_input, d_output, width, height, channels);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaEventRecord(stopKernel));

    // --- D2H transfer ---
    CUDA_CHECK(cudaEventRecord(startD2H));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, totalBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stopD2H));

    CUDA_CHECK(cudaEventRecord(stopTotal));
    CUDA_CHECK(cudaEventSynchronize(stopTotal)); // ceka da SVI event-ovi zavrse

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
    CUDA_CHECK(cudaFree(d_output));

    return result;
}