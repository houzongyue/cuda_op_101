#include "reduce_impls.h"
#include <cuda_runtime.h>

// Kernel: Linear reduction with single thread
__global__ void kernelReduceLinear(int* devArr, int n, int* devRes) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += devArr[i];
    }
    devRes[0] = sum;
}

// Kernel: Parallel reduction with multiple threads v0
__global__ void kernelReduceParallelV0(int* devArr, int n, int* devRes) {
    int threadNum = gridDim.x * blockDim.x;
    int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    int numPerThread = (n + threadNum - 1) / threadNum;
    int start = globalTid * numPerThread;
    int end = start + numPerThread;
    if (end > n) end = n;

    int partialSum = 0;
    for (int i = start; i < end; i++) {
        partialSum += devArr[i];
    }

    // Atomic add to global result
    atomicAdd(devRes, partialSum);
}

// Launcher: GPU Linear reduction
int reduceGpuLinear(int* arr, int n) {
    int* devArr;
    int* devRes;
    int hostRes;

    cudaMalloc(&devArr, sizeof(int) * n);
    cudaMalloc(&devRes, sizeof(int));

    cudaMemcpy(devArr, arr, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemset(devRes, 0, sizeof(int));

    kernelReduceLinear<<<1, 1>>>(devArr, n, devRes);

    cudaMemcpy(&hostRes, devRes, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(devArr);
    cudaFree(devRes);

    return hostRes;
}

// Launcher: GPU Parallel reduction v0
int reduceGpuParallelV0(int* arr, int n) {
    int* devArr;
    int* devRes;
    int hostRes;

    cudaMalloc(&devArr, sizeof(int) * n);
    cudaMalloc(&devRes, sizeof(int));

    cudaMemcpy(devArr, arr, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemset(devRes, 0, sizeof(int));

    // Use reasonable grid/block configuration
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    // Limit grid size to avoid too many blocks
    if (gridSize > 1024) gridSize = 1024;

    kernelReduceParallelV0<<<gridSize, blockSize>>>(devArr, n, devRes);

    cudaMemcpy(&hostRes, devRes, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(devArr);
    cudaFree(devRes);

    return hostRes;
}
