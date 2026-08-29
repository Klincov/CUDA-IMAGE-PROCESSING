#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            throw std::runtime_error(                                            \
                std::string("CUDA error at ") + __FILE__ + ":" +                  \
                std::to_string(__LINE__) + " - " + cudaGetErrorString(err));      \
        }                                                                         \
    } while (0)

#define CUDA_CHECK_LAST_ERROR()                                                   \
    do {                                                                          \
        cudaError_t err = cudaGetLastError();                                     \
        if (err != cudaSuccess) {                                                 \
            throw std::runtime_error(                                            \
                std::string("CUDA kernel launch error at ") + __FILE__ + ":" +    \
                std::to_string(__LINE__) + " - " + cudaGetErrorString(err));      \
        }                                                                         \
    } while (0)