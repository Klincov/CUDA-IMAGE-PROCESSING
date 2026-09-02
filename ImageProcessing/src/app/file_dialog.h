#pragma once
#include <opencv2/opencv.hpp>

// Otvara Windows "Open File" dijalog i vraca izabranu putanju.
// Vraceni pokazivac je validan samo do sledeceg poziva (staticki bafer).
char* browsePath();

// Otvara dijalog za izbor fajla i ucitava sliku sa te putanje preko OpenCV-a.
cv::Mat loadImage();

// Cuva sliku na fiksnu izlaznu putanju (OUTPUT_PATH + "\\output.jpeg").
// Vraca 0 pri uspehu, baca izuzetak pri gresci.
int saveImage(const cv::Mat& image);