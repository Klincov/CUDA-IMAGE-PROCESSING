#pragma once
#include "core/IFilter.h"
#include "core/ICPUTimed.h"
#include "core/CPUTiming.h"

#define GRAY_LEVELS 256

class HistogramEqualizationOMP : public IFilter, public ICPUTimed {
public:
    cv::Mat apply(const cv::Mat& input) override {

        cv::Mat inputGray;
        cv::cvtColor(input, inputGray, cv::COLOR_BGR2GRAY);
        cv::Mat output = cv::Mat(inputGray.size(),inputGray.type());

        int levelsCumSum[GRAY_LEVELS] = { 0 };
        
        CalculateCumSum(inputGray, levelsCumSum);

        int cumSumMin = firstGreaterThanZero(levelsCumSum);

        #pragma omp parallel for
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

        int levelsCumSum[GRAY_LEVELS] = { 0 };

        CalculateCumSum(inputGray, levelsCumSum);

        int cumSumMin = firstGreaterThanZero(levelsCumSum);

#pragma omp parallel for
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
        double end = omp_get_wtime();

        CpuTimingResult result;
        result.totalMs = (end - start) * 1000.0;
        return result;
    }

    std::string name() const override { return "Histogram Equalization (OpenMP)"; }
private:
    void CalculateCumSum(const cv::Mat& image, int *cumSum) {
        #pragma omp parallel
        {
            int localHist[GRAY_LEVELS] = { 0 };

            #pragma omp for
            for (int row = 0; row < image.rows; row++) {
                for (int column = 0; column < image.cols; column++) {
                    localHist[image.at<uchar>(row, column)]++;
                }
            }

            #pragma omp critical
            {
                for (int i = 0; i < GRAY_LEVELS; i++)
                    cumSum[i] += localHist[i];
            }
        }
        for (int i = 1;i < GRAY_LEVELS;i++) {
            cumSum[i] += cumSum[i - 1];
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