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

// Pokrece pun benchmark:5 filtera x N slika x M run-ova za ompScaling=0, 
// ako je ompScaling = true, samo pokrece omp implementacije i to 
// za 2,4,6,8,10,12 threadCount, upisuje rezultate u CSV fajl na datoj putanji.
void runFullBenchmark(const std::string& csvOutputPath,
    int imagesPerResolution = 10,
    int runsPerImage = 5, bool ompScaling = false);