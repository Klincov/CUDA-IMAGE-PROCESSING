#pragma once
#include "core/IFilter.h"
#include "unsharp_kernel.cuh"



class UnsharpMaskingFilterCUDA : public IFilter {
public:
    cv::Mat apply(const cv::Mat& input) override {
        cv::Mat continuousInput = input.isContinuous() ? input : input.clone();

        cv::Mat output(input.size(), input.type());

        launchUnsharpMaskingKernel(
            continuousInput.data,
            output.data,
            input.cols,
            input.rows,
            input.channels(),
            2
        );

        return output;
    }

    std::string name() const override { return "Unsharp Masking (CUDA)"; }
};
