#include "benchmark.h"

#include "core/CUDATiming.cuh"
#include "core/CPUTiming.h"

#include "filters/sequential/InvertFilterSeq.h"
#include "filters/openmp/InvertFilterOMP.h"
#include "filters/cuda/InvertFilterCUDA.h"

#include "filters/sequential/GaussianBlurFilterSeq.h"
#include "filters/openmp/GaussianBlurFilterOMP.h"
#include "filters/cuda/GaussianBlurFilterCUDA.h"

#include "filters/sequential/SobelFilterSeq.h"
#include "filters/openmp/SobelFilterOMP.h"
#include "filters/cuda/SobelFilterCUDA.h"

#include "filters/sequential/UnsharpMaskingFilterSeq.h"
#include "filters/openmp/UnsharpMaskingFilterOMP.h"
#include "filters/cuda/UnsharpMaskingFilterCUDA.h"

#include "filters/sequential/HistogramEqualizationSeq.h"
#include "filters/openmp/HistogramEqualizationOMP.h"
#include "filters/cuda/HistogramEqualizationCUDA.h"

#include <iostream>

void GPUWarmUp() {
    std::cout << "GPU Warm up..." << std::endl << std::endl;
    cv::Mat dummyImage(2000, 2000, CV_8UC3);
    cv::randu(dummyImage, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    cv::Mat dummyOut;
    InvertFilterCUDA().applyTimed(dummyImage, dummyOut);
    GaussianBlurFilterCUDA().applyTimed(dummyImage, dummyOut);
    HistogramEqualizationCUDA().applyTimed(dummyImage, dummyOut);
    SobelFilterCUDA().applyTimed(dummyImage, dummyOut);
    UnsharpMaskingFilterCUDA().applyTimed(dummyImage, dummyOut);
    std::cout << "DONE!" << std::endl << std::endl;
}

bool validationTest(const cv::Mat& seq, const cv::Mat& other) {
    if (seq.type() != other.type()) {
        std::cout << "Image type mismatch!" << std::endl;
        return 0;
    }
    double diff = cv::norm(seq, other, cv::NORM_INF);
    return diff == 0;
}

void filterTimedTest(const cv::Mat& image) {

    GPUWarmUp();
    cv::Mat output = cv::Mat(image.size(), image.type());

    CpuTimingResult resultSeq;
    CpuTimingResult resultOMP;
    CudaTimingResult resultCUDA;

    std::cout << "-----SEQUENTIAL-----" << std::endl;

    resultSeq = InvertFilterSeq().applyTimed(image, output);
    std::cout << "Invert" << std::endl;
    std::cout << "TOTAL: " << resultSeq.totalMs << std::endl << std::endl;

    resultSeq = GaussianBlurFilterSeq().applyTimed(image, output);
    std::cout << "Gauss" << std::endl;
    std::cout << "TOTAL: " << resultSeq.totalMs << std::endl << std::endl;

    resultSeq = HistogramEqualizationSeq().applyTimed(image, output);
    std::cout << "HistEqual" << std::endl;
    std::cout << "TOTAL: " << resultSeq.totalMs << std::endl << std::endl;

    resultSeq = SobelFilterSeq().applyTimed(image, output);
    std::cout << "Sobel" << std::endl;
    std::cout << "TOTAL: " << resultSeq.totalMs << std::endl << std::endl;

    resultSeq = UnsharpMaskingFilterSeq().applyTimed(image, output);
    std::cout << "Unsharp" << std::endl;
    std::cout << "TOTAL: " << resultSeq.totalMs << std::endl << std::endl;

    std::cout << "-----OMP-----" << std::endl;

    resultOMP = InvertFilterOMP().applyTimed(image, output);
    std::cout << "Invert" << std::endl;
    std::cout << "TOTAL: " << resultOMP.totalMs << std::endl << std::endl;

    resultOMP = GaussianBlurFilterOMP().applyTimed(image, output);
    std::cout << "Gauss" << std::endl;
    std::cout << "TOTAL: " << resultOMP.totalMs << std::endl << std::endl;

    resultOMP = HistogramEqualizationOMP().applyTimed(image, output);
    std::cout << "HistEqual" << std::endl;
    std::cout << "TOTAL: " << resultOMP.totalMs << std::endl << std::endl;

    resultOMP = SobelFilterOMP().applyTimed(image, output);
    std::cout << "Sobel" << std::endl;
    std::cout << "TOTAL: " << resultOMP.totalMs << std::endl << std::endl;

    resultOMP = UnsharpMaskingFilterOMP().applyTimed(image, output);
    std::cout << "Unsharp" << std::endl;
    std::cout << "TOTAL: " << resultOMP.totalMs << std::endl << std::endl;

    std::cout << "-----CUDA-----" << std::endl;

    resultCUDA = InvertFilterCUDA().applyTimed(image, output);
    std::cout << "Invert" << std::endl;
    std::cout << "H2D: " << resultCUDA.h2dMs << std::endl;
    std::cout << "kernel: " << resultCUDA.kernelMs << std::endl;
    std::cout << "D2H: " << resultCUDA.d2hMs << std::endl;
    std::cout << "TOTAL: " << resultCUDA.totalMs << std::endl << std::endl;

    resultCUDA = GaussianBlurFilterCUDA().applyTimed(image, output);
    std::cout << "Gauss" << std::endl;
    std::cout << "H2D: " << resultCUDA.h2dMs << std::endl;
    std::cout << "kernel: " << resultCUDA.kernelMs << std::endl;
    std::cout << "D2H: " << resultCUDA.d2hMs << std::endl;
    std::cout << "TOTAL: " << resultCUDA.totalMs << std::endl << std::endl;

    resultCUDA = HistogramEqualizationCUDA().applyTimed(image, output);
    std::cout << "HistEqual" << std::endl;
    std::cout << "H2D: " << resultCUDA.h2dMs << std::endl;
    std::cout << "kernel: " << resultCUDA.kernelMs << std::endl;
    std::cout << "D2H: " << resultCUDA.d2hMs << std::endl;
    std::cout << "TOTAL: " << resultCUDA.totalMs << std::endl << std::endl;

    resultCUDA = SobelFilterCUDA().applyTimed(image, output);
    std::cout << "Sobel" << std::endl;
    std::cout << "H2D: " << resultCUDA.h2dMs << std::endl;
    std::cout << "kernel: " << resultCUDA.kernelMs << std::endl;
    std::cout << "D2H: " << resultCUDA.d2hMs << std::endl;
    std::cout << "TOTAL: " << resultCUDA.totalMs << std::endl << std::endl;

    resultCUDA = UnsharpMaskingFilterCUDA().applyTimed(image, output);
    std::cout << "Unsharp" << std::endl;
    std::cout << "H2D: " << resultCUDA.h2dMs << std::endl;
    std::cout << "kernel: " << resultCUDA.kernelMs << std::endl;
    std::cout << "D2H: " << resultCUDA.d2hMs << std::endl;
    std::cout << "TOTAL: " << resultCUDA.totalMs << std::endl << std::endl;

}