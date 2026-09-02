#include "file_dialog.h"
#include "image_io.h"

#include <windows.h>
#include <commdlg.h>
#include <cstring>
#include <iostream>

#define OUTPUT_PATH "C:\\Users\\mihaj\\OneDrive\\Desktop"

char* browsePath() {
    static OPENFILENAME ofn;
    static char szFile[260] = { 0 };

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

    if (GetOpenFileName(&ofn) == TRUE) {
        std::cout << "Selected file: " << ofn.lpstrFile << std::endl;
    }
    else {
        std::cout << "Browser canceled or failed." << std::endl;
    }
    return szFile;
}

cv::Mat loadImage() {
    try {
        char path[200];
        strcpy(path, browsePath());
        std::cout << path;
        return loadImageCV(path);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << "FAILED!" << std::endl;
        throw;
    }
}

int saveImage(const cv::Mat& image) {
    try {
        char output[300] = { 0 };
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