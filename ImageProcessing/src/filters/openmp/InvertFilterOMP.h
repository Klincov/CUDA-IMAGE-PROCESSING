#pragma once
#include "core/IFilter.h"
#include "core/ICPUTimed.h"
#include "core/CPUTiming.h"
#include <omp.h>

class InvertFilterOMP : public IFilter, public ICPUTimed {
public:
    cv::Mat apply(const cv::Mat& input) override {
        cv::Mat output = cv::Mat(input.size(), input.type());

        // collapse(2) spaja obe petlje u jedan iteracioni prostor,
        // sto daje bolju raspodelu posla po nitima kod manjih slika.
        #pragma omp parallel for collapse(2) schedule(static)
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
        if (numThreads > 0) {
            omp_set_num_threads(numThreads);
        }
        // ako je numThreads == 0, koristi se OMP default (obicno svi dostupni)

        double start = omp_get_wtime();
#pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < input.rows; y++) {
            for (int x = 0; x < input.cols; x++) {
                cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);
                output.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    255 - pixel[0], 255 - pixel[1], 255 - pixel[2]
                );
            }
        }
        double end = omp_get_wtime();

        CpuTimingResult result;
        result.totalMs = (end - start) * 1000.0;
        return result;
    }

    std::string name() const override { return "Invert (OpenMP)"; }
};
