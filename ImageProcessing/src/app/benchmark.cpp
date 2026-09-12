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
#include <fstream>
#include <functional>
#include <vector>
#include <string>

class CsvWriter {
public:
    explicit CsvWriter(const std::string& path) : file(path) {
        if (!file.is_open()) {
            throw std::runtime_error("Can't open CSV write file: " + path);
        }
        file << "filter,implementation,resolution,image_id,run_index,"
            "h2d_ms,kernel_ms,d2h_ms,total_ms,omp_threads,status\n";
    }

    void writeCudaRow(const std::string& filterName, const std::string& resolution,
        int imageId, int runIndex, const CudaTimingResult& r) {
        file << filterName << ",CUDA," << resolution << "," << imageId << "," << runIndex << ","
            << r.h2dMs << "," << r.kernelMs << "," << r.d2hMs << "," << r.totalMs << ",,OK\n";
    }

    void writeCpuRow(const std::string& filterName, const std::string& implementation,
        const std::string& resolution, int imageId, int runIndex,
        const CpuTimingResult& r, int ompThreads = -1) {
        file << filterName << "," << implementation << "," << resolution << "," << imageId << ","
            << runIndex << ",,,," << r.totalMs << ","
            << (ompThreads >= 0 ? std::to_string(ompThreads) : "") << ",OK\n";
    }

    void writeFailedRow(const std::string& filterName, const std::string& implementation,
        const std::string& resolution, int imageId, int runIndex,
        const std::string& errorMsg) {
        file << filterName << "," << implementation << "," << resolution << "," << imageId << ","
            << runIndex << ",,,,," << ",FAILED: " << sanitize(errorMsg) << "\n";
    }

private:
    std::ofstream file;

    // Uklanja zareze/novi red iz poruke greske da ne pokvari CSV strukturu.
    static std::string sanitize(std::string s) {
        for (char& c : s) {
            if (c == ',' || c == '\n' || c == '\r') c = ' ';
        }
        return s;
    }
};

static std::vector<cv::Mat> generateTestImages(int width, int height, int count) {
    std::vector<cv::Mat> images;
    images.reserve(count);
    for (int i = 0; i < count; i++) {
        cv::Mat img(height, width, CV_8UC3);
        cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        images.push_back(img);
    }
    return images;
}

// Definicija jednog filtera za benchmark: ime + lambde koje pozivaju applyTimed

struct FilterBenchmarkEntry {
    std::string name;
    std::function<CpuTimingResult(const cv::Mat&, cv::Mat&)> seqFn;
    std::function<CpuTimingResult(const cv::Mat&, cv::Mat&, const int&)> ompFn;
    std::function<CudaTimingResult(const cv::Mat&, cv::Mat&)> cudaFn;
};

static std::vector<FilterBenchmarkEntry> buildFilterEntries() {
    std::vector<FilterBenchmarkEntry> entries;

    entries.push_back({
        "Invert",
        [](const cv::Mat& in, cv::Mat& out) { return InvertFilterSeq().applyTimed(in, out); },
        [](const cv::Mat& in, cv::Mat& out, const int& num) { return InvertFilterOMP().applyTimed(in, out, num); },
        [](const cv::Mat& in, cv::Mat& out) { return InvertFilterCUDA().applyTimed(in, out); }
        });

    entries.push_back({
        "Gauss",
        [](const cv::Mat& in, cv::Mat& out) { return GaussianBlurFilterSeq().applyTimed(in, out); },
        [](const cv::Mat& in, cv::Mat& out, const int& num) { return GaussianBlurFilterOMP().applyTimed(in, out, num); },
        [](const cv::Mat& in, cv::Mat& out) { return GaussianBlurFilterCUDA().applyTimed(in, out); }
        });

    entries.push_back({
        "Sobel",
        [](const cv::Mat& in, cv::Mat& out) { return SobelFilterSeq().applyTimed(in, out); },
        [](const cv::Mat& in, cv::Mat& out, const int& num) { return SobelFilterOMP().applyTimed(in, out, num); },
        [](const cv::Mat& in, cv::Mat& out) { return SobelFilterCUDA().applyTimed(in, out); }
        });

    entries.push_back({
        "Unsharp",
        [](const cv::Mat& in, cv::Mat& out) { return UnsharpMaskingFilterSeq().applyTimed(in, out); },
        [](const cv::Mat& in, cv::Mat& out, const int& num) { return UnsharpMaskingFilterOMP().applyTimed(in, out, num); },
        [](const cv::Mat& in, cv::Mat& out) { return UnsharpMaskingFilterCUDA().applyTimed(in, out); }
        });

    entries.push_back({
        "HistEqual",
        [](const cv::Mat& in, cv::Mat& out) { return HistogramEqualizationSeq().applyTimed(in, out); },
        [](const cv::Mat& in, cv::Mat& out, const int& num) { return HistogramEqualizationOMP().applyTimed(in, out, num); },
        [](const cv::Mat& in, cv::Mat& out) { return HistogramEqualizationCUDA().applyTimed(in, out); }
        });

    return entries;
}

struct BenchResolution { std::string name; int width; int height; };

void runFullBenchmark(const std::string& csvOutputPath, int imagesPerResolution, int runsPerImage, bool ompScaling) {

    std::vector<BenchResolution> resolutions = {
        {"720p",  1280, 720},
        {"1080p", 1920, 1080},
        {"1440p", 2560, 1440},
        {"2160p", 3840, 2160}
    };
    std::vector<int> ompThreadNums;
    if (ompScaling) {
        ompThreadNums.push_back(2);
        ompThreadNums.push_back(4);
        ompThreadNums.push_back(6);
        ompThreadNums.push_back(8);
        ompThreadNums.push_back(10);
    }
    ompThreadNums.push_back(12);


    auto filterEntries = buildFilterEntries();

    CsvWriter csv(csvOutputPath);

    std::cout << "Starting benchmark. Results in: " << csvOutputPath << std::endl;

    GPUWarmUp(); // JEDNOM

    for (const auto& res : resolutions) {
        std::cout << "\n=== Resolution: " << res.name << " (" << res.width << "x" << res.height << ") ===" << std::endl;

        auto images = generateTestImages(res.width, res.height, imagesPerResolution);

        for (int imgId = 0; imgId < static_cast<int>(images.size()); imgId++) {
            const cv::Mat& image = images[imgId];
            std::cout << std::endl << "------ Image " << (imgId + 1) << "/" << images.size() << "------" << std::endl;

            for (const auto& entry : filterEntries) {
                std::cout << entry.name << std::endl;
                for (int run = 0; run < runsPerImage; run++) {
                    std::cout << "Run " << run + 1 << ": ";
                    cv::Mat output;

                    // --- Sequential ---
                    if (ompScaling) {
                        std::cout << "Sequential...";
                        try {
                            CpuTimingResult r = entry.seqFn(image, output);
                            csv.writeCpuRow(entry.name, "Sequential", res.name, imgId, run, r);
                        }
                        catch (const std::exception& e) {
                            std::cerr << "    FAILED (Sequential/" << entry.name << "): " << e.what() << std::endl;
                            csv.writeFailedRow(entry.name, "Sequential", res.name, imgId, run, e.what());
                        }
                        std::cout << " Done! ";
                    }

                    // --- OMP ---
                    std::cout << "OMP ";
                    for (const auto& threadCount : ompThreadNums) {
                        try {
                            std::cout << threadCount <<"...";
                            CpuTimingResult r = entry.ompFn(image, output, threadCount);
                            csv.writeCpuRow(entry.name, "OpenMP", res.name, imgId, run, r, threadCount);
                        }
                        catch (const std::exception& e) {
                            std::cerr << "    FAILED (OpenMP/" << entry.name << "): " << e.what() << std::endl;
                            csv.writeFailedRow(entry.name, "OpenMP", res.name, imgId, run, e.what());
                        }
                    }
                    std::cout << " Done! ";


                    // --- CUDA ---
                    if (ompScaling) {
                        std::cout << "CUDA...";
                        try {
                            CudaTimingResult r = entry.cudaFn(image, output);
                            csv.writeCudaRow(entry.name, res.name, imgId, run, r);
                        }
                        catch (const std::exception& e) {
                            std::cerr << "    FAILED (CUDA/" << entry.name << "): " << e.what() << std::endl;
                            csv.writeFailedRow(entry.name, "CUDA", res.name, imgId, run, e.what());
                        }
                        std::cout << " Done! " << std::endl;
                    }

                }
            }
        }
    }

    std::cout << "\nBenchmark finished. Results saved in: " << csvOutputPath << std::endl;
}

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