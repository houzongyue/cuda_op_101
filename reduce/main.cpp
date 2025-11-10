#include <iostream>
#include <iomanip>
#include <cstdlib>
#include "timer.h"
#include "reduce_impls.h"

using namespace std;

void printHeader() {
    cout << "\n";
    cout << string(70, '=') << "\n";
    cout << "Array Reduction Benchmark\n";
    cout << string(70, '=') << "\n";
}

void printResult(const string& name, int result, int expected, float timeUs) {
    string status = (result == expected) ? "[PASS]" : "[FAIL]";
    cout << left << setw(25) << name
         << " | Result: " << setw(12) << result
         << " | Time: " << fixed << setprecision(2) << setw(10) << timeUs << " us"
         << " | " << status << "\n";
}

int main() {
    // Configuration
    const int n = 25600000;  // Array size

    // Initialize array
    int* arr = new int[n];
    for (int i = 0; i < n; i++) {
        arr[i] = 1;
    }

    printHeader();
    cout << "Array size: " << n << " elements\n";
    cout << string(70, '-') << "\n";

    // CPU Baseline (also serves as reference)
    int reference;
    float cpuTime;
    {
        CpuTimer timer;
        timer.start();
        reference = reduceCpuBaseline(arr, n);
        timer.stop();
        cpuTime = timer.elapsed();
        printResult("CPU Baseline", reference, reference, cpuTime);
    }

    // GPU Linear
    {
        GpuTimer timer;
        timer.start();
        int result = reduceGpuLinear(arr, n);
        timer.stop();
        printResult("GPU Linear (1 thread)", result, reference, timer.elapsed());
    }

    // GPU Parallel V0
    {
        GpuTimer timer;
        timer.start();
        int result = reduceGpuParallelV0(arr, n);
        timer.stop();
        printResult("GPU Parallel (multi v0)", result, reference, timer.elapsed());
    }

    cout << string(70, '=') << "\n\n";

    delete[] arr;
    return 0;
}
