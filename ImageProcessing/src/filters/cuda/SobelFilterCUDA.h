#pragma once
#include "core/IFilter.h"
#include "sobel_kernel.cuh"
#include "core/ICUDATimed.h"
#include "core/CUDATiming.cuh"

class SobelFilterCUDA : public IFilter, public ICUDATimed {
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

    CudaTimingResult applyTimed(const cv::Mat& input, cv::Mat& output) {
        cv::Mat inputGray;
        cv::cvtColor(input, inputGray, cv::COLOR_BGR2GRAY);

        cv::Mat outputGray;
        outputGray.create(inputGray.size(), inputGray.type());

        try {
            CudaTimingResult r = launchSobelFilterKernelTimed(
                inputGray.data,
                outputGray.data,
                input.cols,
                input.rows
            );
            cv::cvtColor(outputGray, output, cv::COLOR_GRAY2BGR);
            return r;
        }
        catch (const std::exception& e) {
            std::cerr << "Filter failed: " << e.what() << std::endl;
            throw e;
        }

    }

    std::string name() const override { return "Sobel (CUDA)"; }
};
