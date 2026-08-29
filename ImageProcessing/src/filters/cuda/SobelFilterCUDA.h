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

        try{
            launchSobelFilterKernel(inputGray.data, output.data, output.cols, output.rows);
            return output;
        }
        catch (const std::exception& e) {
            std::cerr << "Filter failed: " << e.what() << std::endl;
            throw e;
        }
    }

    std::string name() const override { return "Sobel (CUDA)"; }
};
