#pragma once
#include "core/IFilter.h"
#include "core/ICPUTimed.h"
#include "core/CPUTiming.h"

class GaussianBlurFilterSeq : public IFilter, public ICPUTimed {
public:
    cv::Mat apply(const cv::Mat& input) override {

        cv::Mat output(input.size(), input.type());
        for (int row = 0; row < input.size().height; row++) {
            for (int column = 0; column < input.size().width; column++) {
                calculateNewRGB(input,output, column, row);
            }
        }
        return output;
    }

    CpuTimingResult applyTimed(const cv::Mat& input, cv::Mat& output, int numThreads = 0) override {
        auto start = std::chrono::high_resolution_clock::now();

        output.create(input.size(), input.type());
        for (int row = 0; row < input.size().height; row++) {
            for (int column = 0; column < input.size().width; column++) {
                calculateNewRGB(input, output, column, row);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        CpuTimingResult result;
        result.totalMs = std::chrono::duration<double, std::milli>(end - start).count();

        return result;
    }
    

    std::string name() const override { return "Gaussian Blur (Sequential)"; }
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
                if (targetCol >= input.size().width) targetCol = input.size().width-1;
                if (targetRow >= input.size().height) targetRow = input.size().height-1;

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
