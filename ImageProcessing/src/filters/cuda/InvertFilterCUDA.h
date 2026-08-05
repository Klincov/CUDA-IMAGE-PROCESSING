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

        launchInvertKernel(output.data, output.cols, output.rows, output.channels());
        return output;
    }

    std::string name() const override { return "Invert (CUDA)"; }
};
