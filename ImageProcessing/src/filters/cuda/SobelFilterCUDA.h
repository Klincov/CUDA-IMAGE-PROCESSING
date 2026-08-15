#pragma once
#include "core/IFilter.h"
#include "sobel_kernel.cuh"

class SobelFilterCUDA : public IFilter {
public:
    cv::Mat apply(const cv::Mat& input) override {
        cv::Mat continuousInput = input.isContinuous() ? input : input.clone();
        cv::Mat inputGray;
        cv::cvtColor(continuousInput, inputGray, cv::COLOR_BGR2GRAY);
        cv::Mat output = inputGray.clone();

        launchSobelFilterKernel(inputGray.data, output.data, output.cols, output.rows);
        return output;
    }

    std::string name() const override { return "Sobel (CUDA)"; }
};
