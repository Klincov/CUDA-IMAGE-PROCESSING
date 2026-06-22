#include <iostream>
#include "image_io.h"

int main()
{
    try {
        cv::Mat img = loadImage("images/input/Max.jpeg");
        printImageInfo(img);
        saveImage("images/output/output.jpeg", img);
        std::cout<< "OpenCV works!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

