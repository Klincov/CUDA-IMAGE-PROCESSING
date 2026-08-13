#pragma once
#include "core/IFilter.h"
#include "gauss_kernel.cuh"


class GaussianBlurFilterCUDA : public IFilter {
public:
    cv::Mat apply(const cv::Mat& input) override {
        // cv::Mat.data mora biti continuous da bi indeksiranje (row*width+col)
        // u kernelu bilo validno (bez "rupa" izmedju redova zbog stride-a).
        cv::Mat continuousInput = input.isContinuous() ? input : input.clone();
 
        cv::Mat output(input.size(), input.type());
 
        launchGaussianBlurKernel(
            continuousInput.data,
            output.data,
            input.cols,
            input.rows,
            input.channels()
        );
 
        return output;
    }

    std::string name() const override { return "Gaussian Blur (CUDA)"; }
};
