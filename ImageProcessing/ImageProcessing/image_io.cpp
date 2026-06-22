#include "image_io.h"
#include <iostream>
#include <stdexcept>

cv::Mat loadImage(const std::string& path, bool grayscale) {
    int flag = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
    cv::Mat image = cv::imread(path, flag);

    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + path);
    }

    return image;
}

void saveImage(const std::string& path, const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Cannot save empty image to: " + path);
    }

    bool success = cv::imwrite(path, image);

    if (!success) {
        throw std::runtime_error("Failed to save image to: " + path);
    }
}

void printImageInfo(const cv::Mat& image, const std::string& label) {
    if (!label.empty()) {
        std::cout << "[" << label << "] ";
    }

    std::cout << "Size: " << image.cols << "x" << image.rows
        << " | Channels: " << image.channels()
        << " | Type: " << image.type()
        << std::endl;
}