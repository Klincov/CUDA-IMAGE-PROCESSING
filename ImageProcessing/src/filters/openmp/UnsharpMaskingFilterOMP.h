#pragma once
#include "core/IFilter.h"
#include "filters/openmp/GaussianBlurFilterOMP.h"

class UnsharpMaskingFilterOMP : public IFilter {
public:
    cv::Mat apply(const cv::Mat& input) override {
        cv::Mat blurred = GaussianBlurFilterSeq().apply(input);
        float amount = 2;
        cv::Mat output = input.clone();

        #pragma omp parallel for collapse(2) schedule(static)
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

    std::string name() const override { return "Unsharp Masking (OpenMP)"; }
private:
    uchar clamp(float a) {
        return a > 255 ? 255 : a < 0 ? 0 : a;
    }
};
