#pragma once

// Namerno bez OpenCV i bez CUDA-specificnih tipova u potpisu funkcije -
// ovaj header ukljucuje i obican .cpp fajl (koji ne zna nista o CUDA
// sintaksi), zato mora da bude "cist" C++ potpis.
void launchInvertKernel(unsigned char* h_data, int width, int height, int channels);
