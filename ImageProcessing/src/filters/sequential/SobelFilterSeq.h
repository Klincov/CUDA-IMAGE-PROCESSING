#pragma once
#include "core/IFilter.h"
#include "core/ICPUTimed.h"
#include "core/CPUTiming.h"

class SobelFilterSeq : public IFilter, public ICPUTimed {
public:

    cv::Mat apply(const cv::Mat& input) override {

        cv::Mat inputGray;
        cv::cvtColor(input, inputGray, cv::COLOR_BGR2GRAY);
        cv::Mat output = cv::Mat(input.size(), input.type());

        for (int row = 0; row < input.size().height; row++) {
            for (int column = 0; column < input.size().width; column++) {
                calculateNewValue(inputGray, output, column, row);
            }
        }
        return output;
    }

    CpuTimingResult applyTimed(const cv::Mat& input, cv::Mat& output, int numThreads = 0) override {
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat inputGray;
        cv::cvtColor(input, inputGray, cv::COLOR_BGR2GRAY);
        output.create(inputGray.size(), inputGray.type());

        for (int row = 0; row < input.size().height; row++) {
            for (int column = 0; column < input.size().width; column++) {
                calculateNewValue(inputGray, output, column, row);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        CpuTimingResult result;
        result.totalMs = std::chrono::duration<double, std::milli>(end - start).count();

        return result;
    }

    std::string name() const override { return "Sobel (Sequential)"; }
private:
    const int G_x[3][3] = { -1, 0, 1,
                            -2, 0, 2,
                            -1, 0, 1 };

    const int G_y[3][3] = { -1, -2, -1,
                             0, 0, 0,
                             1, 2, 1 };
    
    void calculateNewValue(const cv::Mat & input, cv::Mat & output,int col, int row) {
        int Gx = 0;
        int Gy = 0;
        for (int i = -1;i <= 1;i++) {
            for (int j = -1; j <= 1;j++) {

                int targetCol = clampInt(col + j, 0, input.cols-1);
                int targetRow = clampInt(row + i, 0, input.rows-1);

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