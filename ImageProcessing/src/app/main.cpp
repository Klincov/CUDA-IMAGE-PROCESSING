#include <iostream>
#include <memory>
#include <vector>
#include <windows.h>
#include <commdlg.h>//file browse
#include "core/IFilter.h"
#include "image_io.h"

#include "core/CUDATiming.cuh"
#include "core/CUDAMaxBandwidth.cuh"

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




#define OUTPUT_PATH "C:\\Users\\mihaj\\OneDrive\\Desktop"

void implementationChoiceMenu(const std::vector<std::unique_ptr<IFilter>> &filters, cv::Mat& image);
bool validationTest(const cv::Mat& seq, const cv::Mat& input);
void printFilterMenu();
void filterMenu(cv::Mat& image);
char* browsePath();
void printMenu();
int saveImage(const cv::Mat& image);
cv::Mat loadImage();

int main()
{
    DisplayPeakBandwidth();
    cv::Mat loadedImage;
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    while (1) {
        try {
            printImageInfo(loadedImage);
            printMenu();

            int input;
            if (!(std::cin >> input)) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid input" << std::endl;
                continue;
            }

            switch (input) {
            case 1:
                if (loadedImage.size().height <= 0)
                    break;
                cv::imshow("Loaded Image", loadedImage);
                cv::waitKey(0);
                cv::destroyAllWindows();
                break;
            case 2:
                loadedImage = loadImage();
                break;
            case 3:
                if (loadedImage.size().height <= 0)
                    break;
                saveImage(loadedImage);
                break;
            case 4: 
                if (loadedImage.size().height <= 0)
                    break;
                filterMenu(loadedImage);
                break;
            case 0:
                return 1;
            default:
                std::cout << "Invalid input" << std::endl;
                break;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    return 0;
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
        cv::Mat output(image.size(), image.type());

        CudaTimingResult result;

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
            result = InvertFilterCUDA().applyTimed(image, output);
            std::cout << "Invert" << std::endl;
            std::cout << "H2D: " << result.h2dMs << std::endl;
            std::cout << "kernel: " << result.kernelMs << std::endl;
            std::cout << "D2H: " << result.d2hMs << std::endl;
            std::cout << "TOTAL: " << result.totalMs << std::endl << std::endl;

            result = GaussianBlurFilterCUDA().applyTimed(image,output);
            std::cout << "Gauss" << std::endl;
            std::cout << "H2D: " << result.h2dMs << std::endl;
            std::cout << "kernel: " << result.kernelMs << std::endl;
            std::cout << "D2H: " << result.d2hMs << std::endl;
            std::cout << "TOTAL: " << result.totalMs << std::endl << std::endl;

            result = HistogramEqualizationCUDA().applyTimed(image, output);
            std::cout << "HistEqual" << std::endl;
            std::cout << "H2D: " << result.h2dMs << std::endl;
            std::cout << "kernel: " << result.kernelMs << std::endl;
            std::cout << "D2H: " << result.d2hMs << std::endl;
            std::cout << "TOTAL: " << result.totalMs << std::endl << std::endl;

            result = SobelFilterCUDA().applyTimed(image, output);
            std::cout << "Sobel" << std::endl;
            std::cout << "H2D: " << result.h2dMs << std::endl;
            std::cout << "kernel: " << result.kernelMs << std::endl;
            std::cout << "D2H: " << result.d2hMs << std::endl;
            std::cout << "TOTAL: " << result.totalMs << std::endl << std::endl;

            result = UnsharpMaskingFilterCUDA().applyTimed(image, output);
            std::cout << "Unsharp" << std::endl;
            std::cout << "H2D: " << result.h2dMs << std::endl;
            std::cout << "kernel: " << result.kernelMs << std::endl;
            std::cout << "D2H: " << result.d2hMs << std::endl;
            std::cout << "TOTAL: " << result.totalMs << std::endl << std::endl;

            
            break;
        case 0:
            return;
        default:
            std::cout << "Invalid input" << std::endl;
            break;
        }
    }
}

void implementationChoiceMenu(const std::vector<std::unique_ptr<IFilter>>& filters, cv::Mat& image){
    std::cout << "\nChoose Implementation:" << std::endl;
    std::cout << "1.Sequential" << std::endl;
    std::cout << "2.OpenMP" << std::endl;
    std::cout << "3.CUDA" << std::endl ;
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

        if (!validationTest(seq,omp)) {
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

bool validationTest(const cv::Mat& seq, const cv::Mat& input) {
    double diff = cv::norm(seq, input, cv::NORM_INF);
    if (diff > 0) return false;
    return true;
}


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
    std::cout << "\t0.Back" << std::endl << std::endl;

    std::cout << "\nInput: " << std::endl;
}

cv::Mat loadImage() {
    try {
        char path[200];
        strcpy(path, browsePath());
        std::cout << path;
        return loadImageCV(path);
        std::cout << "Loaded!" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << "FAILED!" << std::endl;
        throw;
    }
}

int saveImage(const cv::Mat& image) {
    try {
        char output[300] = {0};
        strcat(output, OUTPUT_PATH);
        strcat(output, "\\output.jpeg");
        saveImageCV(output, image);
        std::cout << "Saved!" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << "FAILED!" << std::endl;
        throw;
    }
}

char* browsePath() {
    OPENFILENAME ofn;       // Common dialog box structure
    char szFile[260] = { 0 }; // Buffer for file name

    // Initialize OPENFILENAME
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = "All Files\0*.*\0Text Files\0*.TXT\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

    // Display the Open dialog box 
    if (GetOpenFileName(&ofn) == TRUE) {
        std::cout << "Selected file: " << ofn.lpstrFile << std::endl;
    }
    else {
        std::cout << "Browser canceled or failed." << std::endl;
    }
    return szFile;
}
