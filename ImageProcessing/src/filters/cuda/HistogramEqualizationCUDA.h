#pragma once
#include "core/IFilter.h"
#include "equalization_kernel.cuh"

class HistogramEqualizationCUDA : public IFilter {
public:
    cv::Mat apply(const cv::Mat& input) override {

        cv::Mat inputGray;
        cv::cvtColor(input, inputGray, cv::COLOR_BGR2GRAY);

        cv::Mat output = inputGray.clone();

        launchHistogramEqualizationKernel(
            inputGray.data,
            output.data,
            input.cols,
            input.rows
        );

        return output;
    }

    std::string name() const override { return "Histogram Equalization (CUDA)"; }
};