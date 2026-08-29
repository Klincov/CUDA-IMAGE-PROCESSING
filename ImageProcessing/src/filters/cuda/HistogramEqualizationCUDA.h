#pragma once
#include "core/IFilter.h"
#include "equalization_kernel.cuh"

class HistogramEqualizationCUDA : public IFilter {
public:
    cv::Mat apply(const cv::Mat& input) override {

        cv::Mat inputGray;
        cv::cvtColor(input, inputGray, cv::COLOR_BGR2GRAY);

        cv::Mat output = inputGray.clone();
        try {
            launchHistogramEqualizationKernel(
                inputGray.data,
                output.data,
                input.cols,
                input.rows
            );
        }
        catch (const std::exception& e) {
            std::cerr << "Filter failed: " << e.what() << std::endl;
            throw e;
        }

        return output;
    }

    std::string name() const override { return "Histogram Equalization (CUDA)"; }
};