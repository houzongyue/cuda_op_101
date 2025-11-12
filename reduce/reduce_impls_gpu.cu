#include "reduce_impls.h"
#include "gpu_test_fixture.h"
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
    GpuReduceFixture fixture(arr, n);
    return fixture.execute([n](int* devArr, int arraySize, int* devRes) {
        kernelReduceLinear<<<1, 1>>>(devArr, arraySize, devRes);
    });
}

// Launcher: GPU Parallel reduction v0
int reduceGpuParallelV0(int* arr, int n) {
    GpuReduceFixture fixture(arr, n);
    return fixture.execute([n](int* devArr, int arraySize, int* devRes) {
        // Use reasonable grid/block configuration
        int blockSize = 256;
        int gridSize = (arraySize + blockSize - 1) / blockSize;
        // Limit grid size to avoid too many blocks
        if (gridSize > 1024) gridSize = 1024;

        kernelReduceParallelV0<<<gridSize, blockSize>>>(devArr, arraySize, devRes);
    });
}
