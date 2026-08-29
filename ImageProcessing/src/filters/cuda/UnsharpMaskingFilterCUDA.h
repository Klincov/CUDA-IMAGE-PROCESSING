#pragma once
#include "core/IFilter.h"
#include "unsharp_kernel.cuh"



class UnsharpMaskingFilterCUDA : public IFilter {
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

    std::string name() const override { return "Unsharp Masking (CUDA)"; }
};
