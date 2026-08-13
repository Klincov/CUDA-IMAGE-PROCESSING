#include <iostream>
#include <memory>
#include <vector>
#include <windows.h>
#include <commdlg.h>//file browse
#include "core/IFilter.h"
#include "image_io.h"

#include "filters/sequential/InvertFilterSeq.h"
#include "filters/openmp/InvertFilterOMP.h"
#include "filters/cuda/InvertFilterCUDA.h"

#include "filters/sequential/GaussianBlurFilterSeq.h"
#include "filters/openmp/GaussianBlurFilterOMP.h"
#include "filters/cuda/GaussianBlurFilterCUDA.h"


#define OUTPUT_PATH "C:\\Users\\mihaj\\OneDrive\\Desktop"

void implementationChoiceMenu(const std::vector<std::unique_ptr<IFilter>> &filters, const cv::Mat& image);
void printFilterMenu();
void filterMenu(const cv::Mat& image);
char* browsePath();
void printMenu();
int saveImage(const cv::Mat& image);
cv::Mat loadImage();

int main()
{
    cv::Mat loadedImage;
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    while (1) {
        try {
            printImageInfo(loadedImage);
            printMenu();

            int input;
            std::cin >> input;

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

void filterMenu(const cv::Mat& image) {
    cv::Mat original = image.clone();
    cv::Mat result = image.clone();
    while (1) {
        printImageInfo(original);
        printFilterMenu();
        std::vector<std::unique_ptr<IFilter>> filters;

        int input;
        std::cin >> input;
        switch (input) {
        case 1:
            filters.push_back(std::make_unique<InvertFilterSeq>());
            filters.push_back(std::make_unique<InvertFilterOMP>());
            filters.push_back(std::make_unique<InvertFilterCUDA>());
            implementationChoiceMenu(filters, original);
            break;
        case 2:
            filters.push_back(std::make_unique<GaussianBlurFilterSeq>());
            filters.push_back(std::make_unique<GaussianBlurFilterOMP>());
            filters.push_back(std::make_unique<GaussianBlurFilterCUDA>());
            implementationChoiceMenu(filters, original);
            break;
        case 0:
            return;
        default:
            std::cout << "Invalid input" << std::endl;
            break;
        }
    }
}

void implementationChoiceMenu(const std::vector<std::unique_ptr<IFilter>>& filters, const cv::Mat& image){
    std::cout << "\nChoose Implementation:" << std::endl;
    std::cout << "1.Sequential" << std::endl;
    std::cout << "2.OpenMP" << std::endl;
    std::cout << "3.CUDA" << std::endl << std::endl;

    std::cout << "\nInput: " << std::endl;
    int input;
    std::cin >> input;
    if (input < 1 || input > 3) {
        std::cout << "Invalid input!" << std::endl;
        return;
    }
    cv::imshow("Filter applied!", filters[input - 1]->apply(image));
    cv::waitKey(0);
    cv::destroyAllWindows();
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
