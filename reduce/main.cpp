#include <iostream>
#include <iomanip>
#include <cstdlib>
#include "timer.h"
#include "reduce_impls.h"
#include "benchmark_helper.h"

using namespace std;

int main() {
    // Configuration
    const int n = 25600000;  // Array size

    // Initialize array
    int* arr = new int[n];
    for (int i = 0; i < n; i++) {
        arr[i] = 1;
    }

    printBenchmarkHeader("Array Reduction Benchmark", n);

    // CPU Baseline (also serves as reference)
    auto cpuResult = benchmarkCpu("CPU Baseline", [&]() {
        return reduceCpuBaseline(arr, n);
    }, n);  // Expected result is n (all elements are 1)
    int reference = cpuResult.result;

    // GPU Linear
    benchmarkGpu("GPU Linear (single thread)", [&]() {
        return reduceGpuLinear(arr, n);
    }, reference);

    // GPU Parallel V0
    benchmarkGpu("GPU Parallel (multi v0)", [&]() {
        return reduceGpuParallelV0(arr, n);
    }, reference);

    // GPU Parallel lab
    benchmarkGpu("GPU Parallel (multi lab)", [&]() {
        return reduceGpuParallelLab(arr, n);
    }, reference);

    printBenchmarkFooter();

    delete[] arr;
    return 0;
}
