#pragma once

#include <iostream>
#include <iomanip>
#include <functional>
#include <string>
#include "timer.h"

// Benchmark formatting constants
namespace BenchmarkFormat {
    constexpr int TOTAL_WIDTH = 81;
    constexpr int NAME_WIDTH = 30;
    constexpr int RESULT_WIDTH = 9;
    constexpr int TIME_WIDTH = 10;
}

// Benchmark result structure
struct BenchmarkResult {
    int result;
    float timeUs;
    bool passed;

    BenchmarkResult(int r, float t, int expected)
        : result(r), timeUs(t), passed(r == expected) {}
};

// Print separator line
inline void printSeparator(char c = '=') {
    std::cout << std::string(BenchmarkFormat::TOTAL_WIDTH, c) << "\n";
}

// Print benchmark header
inline void printBenchmarkHeader(const std::string& title, int arraySize) {
    std::cout << "\n";
    printSeparator('=');
    std::cout << title << "\n";
    printSeparator('=');
    std::cout << "Array size: " << arraySize << " elements\n";
    printSeparator('-');
}

// Print benchmark footer
inline void printBenchmarkFooter() {
    printSeparator('=');
    std::cout << "\n";
}

// Print benchmark result with consistent formatting
inline void printBenchmarkResult(const std::string& name, const BenchmarkResult& bench) {
    std::string status = bench.passed ? "[PASS]" : "[FAIL]";
    std::cout << std::left << std::setw(BenchmarkFormat::NAME_WIDTH) << name
              << " | Result: " << std::setw(BenchmarkFormat::RESULT_WIDTH) << bench.result
              << " | Time: " << std::fixed << std::setprecision(2)
              << std::setw(BenchmarkFormat::TIME_WIDTH) << bench.timeUs << " us"
              << " | " << status << "\n";
}

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
        printBenchmarkResult(name, bench);
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
        printBenchmarkResult(name, bench);
    }

    return bench;
}
