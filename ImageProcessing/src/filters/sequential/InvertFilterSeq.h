#pragma once
#include "core/IFilter.h"

class InvertFilterSeq : public IFilter {
public:
    cv::Mat apply(const cv::Mat& input) override {
        cv::Mat output = input.clone();
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

    std::string name() const override { return "Invert (Sequential)"; }
};
