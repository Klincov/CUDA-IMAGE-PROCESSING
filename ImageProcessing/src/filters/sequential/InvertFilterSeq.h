#pragma once
#include "core/IFilter.h"
#include "core/ICPUTimed.h"
#include "core/CPUTiming.h"

class InvertFilterSeq : public IFilter, public ICPUTimed {
public:
    cv::Mat apply(const cv::Mat& input) override {
        cv::Mat output = cv::Mat(input.size(), input.type());
        for (int y = 0; y < input.rows; y++) {
            for (int x = 0; x < input.cols; x++) {
                cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);
                output.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    255 - pixel[0], 255 - pixel[1], 255 - pixel[2]
                );
            }
        }
        return output;
    }

    CpuTimingResult applyTimed(const cv::Mat& input, cv::Mat& output, int numThreads = 0) override {
        auto start = std::chrono::high_resolution_clock::now();

        output.create(input.size(), input.type());
        for (int y = 0; y < input.rows; y++) {
            for (int x = 0; x < input.cols; x++) {
                cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);
                output.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    255 - pixel[0], 255 - pixel[1], 255 - pixel[2]
                );
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        CpuTimingResult result;
        result.totalMs = std::chrono::duration<double, std::milli>(end - start).count();

        return result;
    }

    std::string name() const override { return "Invert (Sequential)"; }
};
