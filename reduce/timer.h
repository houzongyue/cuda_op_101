#pragma once

#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

// CPU Timer using std::chrono
class CpuTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;

public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
    }

    // Returns elapsed time in microseconds
    float elapsed() {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return static_cast<float>(duration.count());
    }
};

// GPU Timer using CUDA events
class GpuTimer {
private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

public:
    GpuTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        cudaEventRecord(start_event);
    }

    void stop() {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
    }

    // Returns elapsed time in microseconds
    float elapsed() {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms * 1000.0f; // Convert ms to us
    }
};
