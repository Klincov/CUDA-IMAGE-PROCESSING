#pragma once
#include "core/IFilter.h"
#include "unsharp_kernel.cuh"
#include "core/ICUDATimed.h"
#include "core/CUDATiming.cuh"

class UnsharpMaskingFilterCUDA : public IFilter, public ICUDATimed {
public:
    cv::Mat apply(const cv::Mat& input) override {
        cv::Mat continuousInput = input.isContinuous() ? input : input.clone();

        cv::Mat output(input.size(), input.type());

        try {
            launchUnsharpMaskingKernel(
                continuousInput.data,
                output.data,
                input.cols,
                input.rows,
                input.channels(),
                2
            );
        }
        catch (const std::exception& e) {
            std::cerr << "Filter failed: " << e.what() << std::endl;
            throw e;
        }

        return output;
    }

    CudaTimingResult applyTimed(const cv::Mat& input, cv::Mat& output) {
        cv::Mat continuousInput = input.isContinuous() ? input : input.clone();

        try {
            return launchUnsharpMaskingKernelTimed(
                continuousInput.data,
                output.data,
                input.cols,
                input.rows,
                input.channels(),
                2
            );
        }
        catch (const std::exception& e) {
            std::cerr << "Filter failed: " << e.what() << std::endl;
            throw e;
        }

    }

    std::string name() const override { return "Unsharp Masking (CUDA)"; }
};
