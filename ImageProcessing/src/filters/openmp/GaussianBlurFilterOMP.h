#pragma once
#include "core/IFilter.h"
#include "core/ICPUTimed.h"
#include "core/CPUTiming.h"
#include <omp.h>

class GaussianBlurFilterOMP : public IFilter, public ICPUTimed {
public:
    cv::Mat apply(const cv::Mat& input) override {
        
        cv::Mat output = cv::Mat(input.size(),input.type());

        // collapse(2) spaja obe petlje u jedan iteracioni prostor,
        // sto daje bolju raspodelu posla po nitima kod manjih slika.
#pragma omp parallel for collapse(2) schedule(static)
        for (int row = 0; row < input.rows; row++) {
            for (int column = 0; column < input.cols; column++) {
                calculateNewRGB(input, output, column, row);
            }
        }
        return output;
    }

    CpuTimingResult applyTimed(const cv::Mat& input, cv::Mat& output, int numThreads = 0) override {
        if (numThreads > 0) {
            omp_set_num_threads(numThreads);
        }
        // ako je numThreads == 0, koristi se OMP default (obicno svi dostupni)
        output.create(input.size(), input.type());

        double start = omp_get_wtime();
#pragma omp parallel for collapse(2) schedule(static)
        for (int row = 0; row < input.rows; row++) {
            for (int column = 0; column < input.cols; column++) {
                calculateNewRGB(input, output, column, row);
            }
        }
        double end = omp_get_wtime();

        CpuTimingResult result;
        result.totalMs = (end - start) * 1000.0;
        return result;
    }

    std::string name() const override { return "Gaussian Blur (OpenMP)"; }
private:
    int kernel[5][5] = { 1,4,7,4,1,
                             4,16,26,16,4,
                             7,26,41,26,7,
                             4,16,26,16,4,
                             1,4,7,4,1 };

    void calculateNewRGB(const cv::Mat& input, cv::Mat& output, int col, int row) {
        int newB = 0;
        int newG = 0;
        int newR = 0;
        for (int i = -2;i <= 2;i++) {
            for (int j = -2; j <= 2;j++) {

                int targetCol = col + j;
                int targetRow = row + i;

                if (targetCol < 0) targetCol = 0;
                if (targetRow < 0) targetRow = 0;
                if (targetCol >= input.size().width) targetCol = input.size().width - 1;
                if (targetRow >= input.size().height) targetRow = input.size().height - 1;

                newB += input.at<cv::Vec3b>(targetRow, targetCol)[0] * kernel[i + 2][j + 2];
                newG += input.at<cv::Vec3b>(targetRow, targetCol)[1] * kernel[i + 2][j + 2];
                newR += input.at<cv::Vec3b>(targetRow, targetCol)[2] * kernel[i + 2][j + 2];
            }
        }
        output.at<cv::Vec3b>(row, col)[0] = newB / 273;
        output.at<cv::Vec3b>(row, col)[1] = newG / 273;
        output.at<cv::Vec3b>(row, col)[2] = newR / 273;
    }
};
