#pragma once
#include "core/IFilter.h"
#include "gauss_kernel.cuh"


class GaussianBlurFilterCUDA : public IFilter {
public:
    cv::Mat apply(const cv::Mat& input) override {
        // cv::Mat.data mora biti continuous da bi indeksiranje (row*width+col)
        // u kernelu bilo validno
        cv::Mat continuousInput = input.isContinuous() ? input : input.clone();
 
        cv::Mat output(input.size(), input.type());
 
        try{
            launchGaussianBlurKernel(
                continuousInput.data,
                output.data,
                input.cols,
                input.rows,
                input.channels()
            );
        }
        catch (const std::exception& e) {
            std::cerr << "Filter failed: " << e.what() << std::endl;
            throw e;
        }
 
        return output;
    }

    std::string name() const override { return "Gaussian Blur (CUDA)"; }
};
