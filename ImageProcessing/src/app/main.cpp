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

#define OUTPUT_PATH "C:\\Users\\mihaj\\OneDrive\\Desktop"

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
                saveImage(loadedImage);
                break;
            case 0:
                return 1;
            default:
                break;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    return 0;
}

void printMenu() {
    std::cout << "Options:" << std::endl;
    std::cout << "\t1.Preview image" << std::endl;
    std::cout << "\t2.Load image" << std::endl;
    std::cout << "\t3.Save image" << std::endl;
    std::cout << "\t4.Filter list" << std::endl << std::endl;
    std::cout << "\t0.Exit" << std::endl;

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
