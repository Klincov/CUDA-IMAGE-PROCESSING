#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>

#include "menu.h"
#include "file_dialog.h"
#include "image_io.h"

#include "core/CUDAMaxBandwidth.cuh"
#include "core/OMPMaxThreads.h"

int main()
{
    DisplayPeakBandwidth();
    DisplayMaxThreads();
    cv::Mat loadedImage(256, 256, CV_8UC3);
    cv::randu(loadedImage, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

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