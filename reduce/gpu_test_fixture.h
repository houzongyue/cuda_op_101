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

    // Run a kernel and return the result
    // Automatically provides devArr, n, and devRes to the kernel
    template<typename KernelFunc>
    int run(dim3 gridSize, dim3 blockSize, KernelFunc kernel) {
        kernel<<<gridSize, blockSize>>>(devArr, n, devRes);
        cudaDeviceSynchronize();

        int result;
        cudaMemcpy(&result, devRes, sizeof(int), cudaMemcpyDeviceToHost);
        return result;
    }

    // Getters for advanced usage if needed
    int* getDevArr() { return devArr; }
    int* getDevRes() { return devRes; }
    int getN() { return n; }
};
