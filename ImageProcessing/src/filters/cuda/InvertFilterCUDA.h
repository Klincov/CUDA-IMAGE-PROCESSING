#pragma once
#include "core/IFilter.h"
#include "invert_kernel.cuh"


class InvertFilterCUDA : public IFilter {
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

    std::string name() const override { return "Invert (CUDA)"; }
};
