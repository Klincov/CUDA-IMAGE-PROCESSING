#include "menu.h"
#include "benchmark.h"

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

#include "image_io.h"

#include <iostream>
#include <limits>

void printMenu() {
    std::cout << "Options:" << std::endl;
    std::cout << "\t1.Preview image" << std::endl;
    std::cout << "\t2.Load image" << std::endl;
    std::cout << "\t3.Save image" << std::endl;
    std::cout << "\t4.Filter test" << std::endl;
    std::cout << "\t5.Run benchmark" << std::endl;
    std::cout << "\t0.Exit" << std::endl << std::endl;

    std::cout << "\nInput: " << std::endl;
}

void printFilterMenu() {
    std::cout << "Options:" << std::endl;
    std::cout << "\t1.Invert color" << std::endl;
    std::cout << "\t2.Gaussian blur" << std::endl;
    std::cout << "\t3.Sobel filter" << std::endl;
    std::cout << "\t4.Unsharp masking filter" << std::endl;
    std::cout << "\t5.Histogram Equalization" << std::endl;
    std::cout << "\t6.All filter timed test" << std::endl;
    std::cout << "\t0.Back" << std::endl << std::endl;

    std::cout << "\nInput: " << std::endl;
}

void implementationChoiceMenu(const std::vector<std::unique_ptr<IFilter>>& filters, cv::Mat& image) {
    std::cout << "\nChoose Implementation:" << std::endl;
    std::cout << "1.Sequential" << std::endl;
    std::cout << "2.OpenMP" << std::endl;
    std::cout << "3.CUDA" << std::endl;
    std::cout << "4.Validation" << std::endl << std::endl;

    std::cout << "\nInput: " << std::endl;
    int input;
    std::cin >> input;
    if (input < 1 || input > 4) {
        std::cout << "Invalid input!" << std::endl;
        return;
    }

    if (input == 4) {
        cv::Mat seq = filters[0]->apply(image);
        cv::Mat omp = filters[1]->apply(image);
        cv::Mat cuda = filters[2]->apply(image);

        if (!validationTest(seq, omp)) {
            std::cout << "\nVALIDATION FAILED! OMP DOES NOT MATCH SEQUENTIAL!" << std::endl;
            return;
        }

        if (!validationTest(seq, cuda)) {
            std::cout << "\nVALIDATION FAILED! CUDA DOES NOT MATCH SEQUENTIAL!" << std::endl;
            return;
        }

        std::cout << "\nVALIDATION PASSED!\n" << std::endl;
        return;
    }

    cv::Mat applied = filters[input - 1]->apply(image);
    cv::imshow("Filter applied!", applied);
    cv::waitKey(0);
    cv::destroyAllWindows();

    std::cout << "\nSave?(Y/N)" << std::endl;

    std::cout << "\nInput: " << std::endl;
    char save = 0;
    std::cin >> save;
    if (save == 'Y' || save == 'y') {
        cv::cvtColor(applied, image, cv::COLOR_GRAY2BGR);
    }
    else return;
}

void filterMenu(cv::Mat& image) {
    while (1) {
        printImageInfo(image);
        printFilterMenu();
        std::vector<std::unique_ptr<IFilter>> filters;

        int input;
        if (!(std::cin >> input)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input" << std::endl;
            continue;
        }
        switch (input) {
        case 1:
            filters.push_back(std::make_unique<InvertFilterSeq>());
            filters.push_back(std::make_unique<InvertFilterOMP>());
            filters.push_back(std::make_unique<InvertFilterCUDA>());
            implementationChoiceMenu(filters, image);
            break;
        case 2:
            filters.push_back(std::make_unique<GaussianBlurFilterSeq>());
            filters.push_back(std::make_unique<GaussianBlurFilterOMP>());
            filters.push_back(std::make_unique<GaussianBlurFilterCUDA>());
            implementationChoiceMenu(filters, image);
            break;
        case 3:
            filters.push_back(std::make_unique<SobelFilterSeq>());
            filters.push_back(std::make_unique<SobelFilterOMP>());
            filters.push_back(std::make_unique<SobelFilterCUDA>());
            implementationChoiceMenu(filters, image);
            break;
        case 4:
            filters.push_back(std::make_unique<UnsharpMaskingFilterSeq>());
            filters.push_back(std::make_unique<UnsharpMaskingFilterOMP>());
            filters.push_back(std::make_unique<UnsharpMaskingFilterCUDA>());
            implementationChoiceMenu(filters, image);
            break;
        case 5:
            filters.push_back(std::make_unique<HistogramEqualizationSeq>());
            filters.push_back(std::make_unique<HistogramEqualizationOMP>());
            filters.push_back(std::make_unique<HistogramEqualizationCUDA>());
            implementationChoiceMenu(filters, image);
            break;
        case 6:
            filterTimedTest(image);
            break;
        case 0:
            return;
        default:
            std::cout << "Invalid input" << std::endl;
            break;
        }
    }
}