#pragma once

#include <iostream>
#include <iomanip>
#include <functional>
#include "timer.h"

// Benchmark result structure
struct BenchmarkResult {
    int result;
    float timeUs;
    bool passed;

    BenchmarkResult(int r, float t, int expected)
        : result(r), timeUs(t), passed(r == expected) {}
};

// CPU benchmark helper
inline BenchmarkResult benchmarkCpu(
    const std::string& name,
    std::function<int()> func,
    int expected,
    bool print = true
) {
    CpuTimer timer;
    timer.start();
    int result = func();
    timer.stop();

    BenchmarkResult bench(result, timer.elapsed(), expected);

    if (print) {
        std::string status = bench.passed ? "[PASS]" : "[FAIL]";
        std::cout << std::left << std::setw(25) << name
                  << " | Result: " << std::setw(12) << result
                  << " | Time: " << std::fixed << std::setprecision(2)
                  << std::setw(10) << bench.timeUs << " us"
                  << " | " << status << "\n";
    }

    return bench;
}

// GPU benchmark helper
inline BenchmarkResult benchmarkGpu(
    const std::string& name,
    std::function<int()> func,
    int expected,
    bool print = true
) {
    GpuTimer timer;
    timer.start();
    int result = func();
    timer.stop();

    BenchmarkResult bench(result, timer.elapsed(), expected);

    if (print) {
        std::string status = bench.passed ? "[PASS]" : "[FAIL]";
        std::cout << std::left << std::setw(25) << name
                  << " | Result: " << std::setw(12) << result
                  << " | Time: " << std::fixed << std::setprecision(2)
                  << std::setw(10) << bench.timeUs << " us"
                  << " | " << status << "\n";
    }

    return bench;
}
