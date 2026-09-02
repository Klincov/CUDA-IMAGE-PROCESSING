#pragma once
#include "core/IFilter.h"
#include "filters/sequential/GaussianBlurFilterSeq.h"
#include "core/ICPUTimed.h"
#include "core/CPUTiming.h"

class UnsharpMaskingFilterSeq : public IFilter, public ICPUTimed {
public:
    cv::Mat apply(const cv::Mat& input) override {
        cv::Mat blurred = GaussianBlurFilterSeq().apply(input);
        float amount = 2;
        cv::Mat output = cv::Mat(input.size(),input.type());

        for (int row = 0; row < input.size().height; row++) {
            for (int column = 0; column < input.size().width; column++) {

                int detail_B = input.at<cv::Vec3b>(row, column)[0] - blurred.at<cv::Vec3b>(row, column)[0];
                int detail_G = input.at<cv::Vec3b>(row, column)[1] - blurred.at<cv::Vec3b>(row, column)[1];
                int detail_R = input.at<cv::Vec3b>(row, column)[2] - blurred.at<cv::Vec3b>(row, column)[2];

                output.at<cv::Vec3b>(row, column)[0] = clamp(input.at<cv::Vec3b>(row, column)[0] + amount * detail_B);
                output.at<cv::Vec3b>(row, column)[1] = clamp(input.at<cv::Vec3b>(row, column)[1] + amount * detail_G);
                output.at<cv::Vec3b>(row, column)[2] = clamp(input.at<cv::Vec3b>(row, column)[2] + amount * detail_R);
            }
        }
        return output;
    }

    CpuTimingResult applyTimed(const cv::Mat& input, cv::Mat& output, int numThreads = 0) override {
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat blurred = GaussianBlurFilterSeq().apply(input);
        float amount = 2;
        output.create(input.size(), input.type());

        for (int row = 0; row < input.size().height; row++) {
            for (int column = 0; column < input.size().width; column++) {

                int detail_B = input.at<cv::Vec3b>(row, column)[0] - blurred.at<cv::Vec3b>(row, column)[0];
                int detail_G = input.at<cv::Vec3b>(row, column)[1] - blurred.at<cv::Vec3b>(row, column)[1];
                int detail_R = input.at<cv::Vec3b>(row, column)[2] - blurred.at<cv::Vec3b>(row, column)[2];

                output.at<cv::Vec3b>(row, column)[0] = clamp(input.at<cv::Vec3b>(row, column)[0] + amount * detail_B);
                output.at<cv::Vec3b>(row, column)[1] = clamp(input.at<cv::Vec3b>(row, column)[1] + amount * detail_G);
                output.at<cv::Vec3b>(row, column)[2] = clamp(input.at<cv::Vec3b>(row, column)[2] + amount * detail_R);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        CpuTimingResult result;
        result.totalMs = std::chrono::duration<double, std::milli>(end - start).count();

        return result;
    }

    std::string name() const override { return "Unsharp Masking (Sequential)"; }    
private:
    uchar clamp(float a) {
        return a > 255 ? 255 : a < 0 ? 0 : a ;
    }
};
