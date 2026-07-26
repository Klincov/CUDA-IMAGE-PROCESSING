#include <iostream>
#include "image_io.h"

void printMenu();
int saveImage(const cv::Mat& image);
cv::Mat loadImage();

int main()
{
        cv::Mat loadedImage;
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

        while (1) {
            try{
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

void printMenu(){
    std::cout << "Options:" << std::endl;
    std::cout << "\t1.Preview image" << std::endl;
    std::cout << "\t2.Load image" << std::endl;
    std::cout << "\t3.Save image" << std::endl;
    std::cout << "\t4.Filter list" << std::endl << std::endl;
    std::cout << "\t0.Exit" << std::endl;

    std::cout << "\nInput: " << std::endl;
}

cv::Mat loadImage() {
    std::cout << "Relative path:" << std::endl;
    try {
        std::string input;
        std::cin >> input;
        return loadImageCV(input);
        std::cout << "Loaded!" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << "FAILED!" << std::endl;
        throw;
    }
}

int saveImage(const cv::Mat& image) {
    std::cout << "Relative path:" << std::endl;
    try {
        std::string input;
        std::cin >> input;
        saveImageCV(input, image);
        std::cout << "Saved!" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << "FAILED!" << std::endl;
        throw;
    }
}
