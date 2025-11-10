#pragma once

#include <cuda_runtime.h>

// All reduction implementations share the same signature:
// - arr: input array (host memory)
// - n: array size
// Returns: sum of all elements

// CPU implementations
int reduceCpuBaseline(int* arr, int n);

// GPU implementations (launcher functions handle device memory and configuration)
int reduceGpuLinear(int* arr, int n);
int reduceGpuParallelV0(int* arr, int n);
