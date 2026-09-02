#pragma once
#include <opencv2/opencv.hpp>

// Pokrece jedan GPU kernel po filteru na dummy slici radi zagrevanja
// CUDA konteksta.
// Poziva se JEDNOM na pocetku benchmarka, pre bilo kog merenja.
void GPUWarmUp();

// Pokrece Sequential/OMP/CUDA verziju svih 5 filtera nad datom slikom
// i ispisuje izmerena vremena (H2D/kernel/D2H za CUDA, total za CPU).
void filterTimedTest(const cv::Mat& image);

// Poredi dve slike piksel-po-piksel (NORM_INF razlika).
// Vraca true ako su identicne (diff == 0).
bool validationTest(const cv::Mat& seq, const cv::Mat& other);