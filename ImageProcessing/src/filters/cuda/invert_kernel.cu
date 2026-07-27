#include "invert_kernel.cuh"
#include <cuda_runtime.h>

// __global__ = kernel koji se izvrsava na GPU-u, poziva se sa CPU-a.
// Svaka nit obradjuje tacno jedan piksel (idealno za point-operacije).
__global__ void invertKernel(unsigned char* data, int totalBytes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalBytes) {
        data[idx] = 255 - data[idx];
    }
}

// Ova funkcija je "obican" C++ kod (host kod) koji se kompajlira sa nvcc,
// ali se poziva iz .cpp fajlova kao svaka druga funkcija - to je most
// izmedju CUDA sveta i ostatka aplikacije.
void launchInvertKernel(unsigned char* h_data, int width, int height, int channels) {
    int totalBytes = width * height * channels;
    unsigned char* d_data = nullptr;

    cudaMalloc(&d_data, totalBytes);
    cudaMemcpy(d_data, h_data, totalBytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (totalBytes + threadsPerBlock - 1) / threadsPerBlock;
    invertKernel<<<blocks, threadsPerBlock>>>(d_data, totalBytes);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, totalBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}
