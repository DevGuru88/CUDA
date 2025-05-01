
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// CUDA kernel
__global__ void vectorAdd(float *a, float *b, float *c, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

// Function to check for CUDA errors
void checkCuda(cudaError_t result, const char *func) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", func, cudaGetErrorString(result));
        exit(1);
    }
}

int main() {
    int N = 1 << 20; // 1 million elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c_seq = (float*)malloc(size);
    float *h_c_gpu = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = 8.0f;
        h_b[i] = 8.0f;
    }

    // Sequential computation timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        h_c_seq[i] = h_a[i] + h_b[i];
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end_cpu - start_cpu).count();
    printf("CPU (Sequential) Time: %f seconds\n", cpu_time);

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    checkCuda(cudaMalloc((void**)&d_a, size), "cudaMalloc d_a");
    checkCuda(cudaMalloc((void**)&d_b, size), "cudaMalloc d_b");
    checkCuda(cudaMalloc((void**)&d_c, size), "cudaMalloc d_c");

    // Copy host memory to device
    checkCuda(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice), "cudaMemcpy d_a");
    checkCuda(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice), "cudaMemcpy d_b");

    // Launch kernel and measure GPU time
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    checkCuda(cudaEventRecord(start), "cudaEventRecord start");
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");

    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float milliseconds = 0;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");

    printf("GPU (Parallel) Time: %f milliseconds\n", milliseconds);

    // Copy result back to host
    checkCuda(cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost), "cudaMemcpy d_c");

    // Optional correctness check (compare sequential and GPU results)
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_c_gpu[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c_seq);
    free(h_c_gpu);

    return 0;
}
