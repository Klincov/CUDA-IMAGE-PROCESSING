#pragma once

struct CudaTimingResult {
    float h2dMs = 0.0f;      // Host - Device transfer
    float kernelMs = 0.0f;   // Cisto vreme izvrsavanja kernela (svih faza/kernela zbirno)
    float d2hMs = 0.0f;      // Device - Host transfer
    float totalMs = 0.0f;    // h2d + kernel + d2h 
};
