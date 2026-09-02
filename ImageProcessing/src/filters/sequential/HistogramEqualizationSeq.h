#pragma once
#include "core/IFilter.h"
#include "core/ICPUTimed.h"
#include "core/CPUTiming.h"

#define GRAY_LEVELS 256

class HistogramEqualizationSeq : public IFilter, public ICPUTimed {
public:
    cv::Mat apply(const cv::Mat& input) override {

        cv::Mat inputGray;
        cv::cvtColor(input, inputGray, cv::COLOR_BGR2GRAY);
        cv::Mat output = inputGray.clone();

        int levelsCumSum[GRAY_LEVELS] = { 0 };
        
        CalculateCumSum(inputGray, levelsCumSum);

        int cumSumMin = firstGreaterThanZero(levelsCumSum);

        for (int row = 0; row < input.size().height; row++) {
            for (int column = 0; column < input.size().width; column++) {

                double equalized = round(
                    (
                        (1.0 * levelsCumSum[inputGray.at<uchar>(row, column)] - cumSumMin) /
                        ((input.rows * input.cols) - cumSumMin)
                    ) *
                    (GRAY_LEVELS - 1)
                );
                
                output.at<uchar>(row, column) = static_cast<unsigned char>(equalized);
            }
        }
        cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
        return output;
    }

    CpuTimingResult applyTimed(const cv::Mat& input, cv::Mat& output, int numThreads = 0) override {
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat inputGray;
        cv::cvtColor(input, inputGray, cv::COLOR_BGR2GRAY);
        output.create(inputGray.size(), inputGray.type());

        int levelsCumSum[GRAY_LEVELS] = { 0 };

        CalculateCumSum(inputGray, levelsCumSum);

        int cumSumMin = firstGreaterThanZero(levelsCumSum);

        for (int row = 0; row < input.size().height; row++) {
            for (int column = 0; column < input.size().width; column++) {

                double equalized = round(
                    (
                        (1.0 * levelsCumSum[inputGray.at<uchar>(row, column)] - cumSumMin) /
                        ((input.rows * input.cols) - cumSumMin)
                        ) *
                    (GRAY_LEVELS - 1)
                );

                output.at<uchar>(row, column) = static_cast<unsigned char>(equalized);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        CpuTimingResult result;
        result.totalMs = std::chrono::duration<double, std::milli>(end - start).count();
        cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);

        return result;
    }

    std::string name() const override { return "Histogram Equalization (Sequential)"; }
private:
    void CalculateCumSum(const cv::Mat& image, int *cumSum) {
        for (int row = 0; row < image.rows; row++) {
            for (int column = 0; column < image.cols; column++) {
                cumSum[image.at<uchar>(row, column)] += 1;
            }
        }
        for (int i = 1;i < GRAY_LEVELS;i++) {
            cumSum[i] += cumSum[i-1];
        }
    }

    int firstGreaterThanZero(int* array) {
        for (int i = 0; i < GRAY_LEVELS; i++) {
            if (array[i] > 0) {
                return array[i];
            }
        }
    }
};