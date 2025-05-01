#include <iostream>
#include <cuda_runtime.h>

__global__ void matmul_kernel(float *a, float *b, float *c, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < N && col < N) {
        float value = 0;
        for(int i = 0; i < N; i++) {
            value += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = value;
    }
}

int main() {
    int N = 3; // Matrix size
    float a[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float b[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    float c[9] = {0};  // Result matrix initialized to 0

    // Device pointers
    float *d_a, *d_b, *d_c;

    // Allocate device memory
    cudaMalloc((void **)&d_a, 9 * sizeof(float));
    cudaMalloc((void **)&d_b, 9 * sizeof(float));
    cudaMalloc((void **)&d_c, 9 * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_a, a, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, 9 * sizeof(float), cudaMemcpyHostToDevice);

    // Set up kernel execution configuration
    dim3 block(3, 3);
    dim3 grid(1, 1);

    // Launch the kernel
    matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to host memory
    cudaMemcpy(c, d_c, 9 * sizeof(float), cudaMemcpyDeviceToHost);

    // Check for errors in memory copying
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Print the result matrix
    std::cout << "Matrix multiplication result:\n";
    for (int i = 0; i < 9; i++) {
        std::cout << c[i] << " ";
        if ((i + 1) % N == 0) std::cout << "\n";
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
