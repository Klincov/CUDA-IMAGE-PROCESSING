#pragma once
#include "core/IFilter.h"
#include "histogram_equalization_kernel.cuh"
#include "core/ICUDATimed.h"
#include "core/CUDATiming.cuh"

class HistogramEqualizationCUDA : public IFilter, public ICUDATimed {
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
            cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
        }
        catch (const std::exception& e) {
            std::cerr << "Filter failed: " << e.what() << std::endl;
            throw e;
        }

        return output;
    }

    CudaTimingResult applyTimed(const cv::Mat& input, cv::Mat& output) {
        cv::Mat inputGray;
        cv::cvtColor(input, inputGray, cv::COLOR_BGR2GRAY);

        cv::Mat outputGray;
        outputGray.create(inputGray.size(), inputGray.type());

        try {
            CudaTimingResult r = launchHistogramEqualizationKernelTimed(
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

    std::string name() const override { return "Histogram Equalization (CUDA)"; }
};