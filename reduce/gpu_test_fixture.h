#pragma once

#include <cuda_runtime.h>
#include <functional>

// GPU Test Fixture: Handles common device memory operations for reduction kernels
class GpuReduceFixture {
private:
    int* devArr;
    int* devRes;
    int* hostArr;
    int n;

public:
    GpuReduceFixture(int* arr, int arraySize) : hostArr(arr), n(arraySize) {
        // Allocate device memory
        cudaMalloc(&devArr, sizeof(int) * n);
        cudaMalloc(&devRes, sizeof(int));

        // Copy input array to device
        cudaMemcpy(devArr, hostArr, sizeof(int) * n, cudaMemcpyHostToDevice);

        // Initialize result to 0
        cudaMemset(devRes, 0, sizeof(int));
    }

    ~GpuReduceFixture() {
        cudaFree(devArr);
        cudaFree(devRes);
    }

    // Execute a kernel with given grid and block configuration
    // The kernel function should be a lambda or function that launches the kernel
    int execute(std::function<void(int*, int, int*)> kernelLauncher) {
        // Launch the kernel via the provided launcher
        kernelLauncher(devArr, n, devRes);

        // Copy result back to host
        int hostRes;
        cudaMemcpy(&hostRes, devRes, sizeof(int), cudaMemcpyDeviceToHost);

        return hostRes;
    }

    // Getters for direct kernel access if needed
    int* getDevArr() { return devArr; }
    int* getDevRes() { return devRes; }
    int getN() { return n; }
};
