#pragma once
#include "core/IFilter.h"
#include "core/ICPUTimed.h"
#include "core/CPUTiming.h"
#include <omp.h>

class SobelFilterOMP : public IFilter, public ICPUTimed {
public:

    cv::Mat apply(const cv::Mat& input) override {

        cv::Mat inputGray;
        cv::cvtColor(input, inputGray, cv::COLOR_BGR2GRAY);
        cv::Mat output = cv::Mat(inputGray.size(), inputGray.type());

        #pragma omp parallel for collapse(2) schedule(static)
        for (int row = 0; row < input.size().height; row++) {
            for (int column = 0; column < input.size().width; column++) {
                calculateNewValue(inputGray, output, column, row);
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
        cv::Mat inputGray;
        cv::cvtColor(input, inputGray, cv::COLOR_BGR2GRAY);
        output.create(inputGray.size(), inputGray.type());

#pragma omp parallel for collapse(2) schedule(static)
        for (int row = 0; row < input.size().height; row++) {
            for (int column = 0; column < input.size().width; column++) {
                calculateNewValue(inputGray, output, column, row);
            }
        }
        double end = omp_get_wtime();

        CpuTimingResult result;
        result.totalMs = (end - start) * 1000.0;
        return result;
    }

    std::string name() const override { return "Sobel (OpenMP)"; }
private:
    const int G_x[3][3] = { -1, 0, 1,
                            -2, 0, 2,
                            -1, 0, 1 };

    const int G_y[3][3] = { -1, -2, -1,
                             0, 0, 0,
                             1, 2, 1 };

    void calculateNewValue(const cv::Mat& input, cv::Mat& output, int col, int row) {
        int Gx = 0;
        int Gy = 0;
        for (int i = -1;i <= 1;i++) {
            for (int j = -1; j <= 1;j++) {

                int targetCol = clampInt(col + j, 0, input.cols - 1);
                int targetRow = clampInt(row + i, 0, input.rows - 1);

                Gx += input.at<uchar>(targetRow, targetCol) * G_x[i + 1][j + 1];
                Gy += input.at<uchar>(targetRow, targetCol) * G_y[i + 1][j + 1];

            }
        }
        output.at<uchar>(row, col) = (abs(Gx) + abs(Gy)) >= 255 ? 255 : (abs(Gx) + abs(Gy));
    }

    int abs(int n) {
        if (n < 0) return -n;
        else return n;
    }

    int clampInt(int v, int lo, int hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};