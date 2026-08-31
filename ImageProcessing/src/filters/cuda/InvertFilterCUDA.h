#pragma once
#include "core/IFilter.h"
#include "invert_kernel.cuh"
#include "core/ICUDATimed.h"
#include "core/CUDATiming.cuh"


class InvertFilterCUDA : public IFilter, public ICUDATimed {
public:
    cv::Mat apply(const cv::Mat& input) override {
        cv::Mat output = input.clone();
        if (!output.isContinuous()) {
            output = output.clone();
        }
        try{
            launchInvertKernel(output.data, output.cols, output.rows, output.channels());
            return output;
        }
        catch (const std::exception& e) {
            std::cerr << "Filter failed: " << e.what() << std::endl;
            throw e;
        }
    }

    CudaTimingResult applyTimed(const cv::Mat& input, cv::Mat& output) override {
        cv::Mat continuousInput = input.isContinuous() ? input : input.clone();

        try {
            CudaTimingResult result = launchInvertKernelTimed(
                continuousInput.data,
                input.cols, input.rows, input.channels()
            );
            output = continuousInput.clone();
            return result;
        }
        catch (const std::exception& e) {
            std::cerr << "Filter failed: " << e.what() << std::endl;
            throw e;
        }
    }

    std::string name() const override { return "Invert (CUDA)"; }
};
