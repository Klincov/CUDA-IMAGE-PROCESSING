#pragma once
#include "core/IFilter.h"
#include "gauss_kernel.cuh"
#include "core/ICUDATimed.h"


class GaussianBlurFilterCUDA : public IFilter, public ICUDATimed {
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

    CudaTimingResult applyTimed(const cv::Mat& input, cv::Mat& output) override {
        cv::Mat continuousInput = input.isContinuous() ? input : input.clone();
        output.create(input.size(), input.type());

        return launchGaussianBlurKernelTimed(
            continuousInput.data, output.data,
            input.cols, input.rows, input.channels()
        );
    }

    std::string name() const override { return "Gaussian Blur (CUDA)"; }
};
