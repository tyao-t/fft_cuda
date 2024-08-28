#include <iostream>
#include <cuda_runtime.h>

__global__ void shuffleKernel(int* input, int* output, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        int newIndex = 0, tmp = idx;
        for (;n>=2;n/=2) {
            newIndex += (tmp % 2) * (n/2);
            tmp /= 2;
        }
        output[newIndex] = input[idx];
    }
}

void shuffleArray(int* h_input, int n) {
    int* d_input;
    int* d_output;

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    shuffleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);

    cudaMemcpy(h_input, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    const int n = 16;
    int h_input[n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    std::cout << "Original array: ";
    for (int i = 0; i < n; ++i) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;

    // Shuffle the array
    shuffleArray(h_input, n);

    std::cout << "Shuffled array: ";
    for (int i = 0; i < n; ++i) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
