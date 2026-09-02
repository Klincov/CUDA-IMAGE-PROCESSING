#pragma once
#include "core/IFilter.h"
#include "core/ICPUTimed.h"
#include "core/CPUTiming.h"
#include "filters/openmp/GaussianBlurFilterOMP.h"

class UnsharpMaskingFilterOMP : public IFilter, public ICPUTimed{
public:
    cv::Mat apply(const cv::Mat& input) override {
        cv::Mat blurred = GaussianBlurFilterOMP().apply(input);
        float amount = 2;
        cv::Mat output = cv::Mat(input.size(),input.type());

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

    CpuTimingResult applyTimed(const cv::Mat& input, cv::Mat& output, int numThreads = 0) override {
        if (numThreads > 0) {
            omp_set_num_threads(numThreads);
        }
        // ako je numThreads == 0, koristi se OMP default (obicno svi dostupni)

        double start = omp_get_wtime();
        cv::Mat blurred = cv::Mat(input.size(), input.type());
        GaussianBlurFilterOMP().applyTimed(input,blurred,numThreads);

        float amount = 2;

        output.create(input.size(),input.type());

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

        double end = omp_get_wtime();

        CpuTimingResult result;
        result.totalMs = (end - start) * 1000.0;
        return result;
    }

    std::string name() const override { return "Unsharp Masking (OpenMP)"; }
private:
    uchar clamp(float a) {
        return a > 255 ? 255 : a < 0 ? 0 : a;
    }
};
