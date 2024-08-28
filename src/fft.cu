#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
// #include "CUDAUniquePtr.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <string>
#include <memory>
#include <cctype>

#ifdef ENABLE_DEBUG
#define DEBUG_LOG(arr, n) \
      do { \
        cuDoubleComplex res[n]; \
        cudaMemcpy(res, arr, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost); \
        for (int i = 0; i < n; i++) { \
            std::cout << cuCreal(res[i]) << " + " << cuCimag(res[i]) << "j" << std::endl; \
        } \
        std::cout << std::endl; \
        cudaDeviceSynchronize(); \
      } while (0);
#else
#define DEBUG_LOG(arr, n)
#endif

// using namespace std;
using std::vector, std::cout, std::endl, std::string, std::ostringstream, std::reverse, std::round, std::isdigit;

__global__ void __attribute__((unused)) computeRootsOfUnity(cuDoubleComplex* omega, int n) {
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    if (k < n) {
        double angle = -2.0 * M_PI * k / n;
        omega[k] = make_cuDoubleComplex(cos(angle), sin(angle));
    }
}

__global__ void __attribute__((unused)) computeConjugate(cuDoubleComplex* arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        arr[idx] = cuConj(arr[idx]);
    }
}

__global__ void hadamargProduct(cuDoubleComplex* a, cuDoubleComplex* b, cuDoubleComplex* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = cuCmul(a[idx], b[idx]);
    }
}

__global__ void shuffle(cuDoubleComplex* input, cuDoubleComplex* output, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        int newIndex = 0, tmp = idx;
        for (; n>=2; n/=2) {
            newIndex += (tmp % 2) * (n/2);
            tmp /= 2;
        }
        output[newIndex] = input[idx];
    }
}

__global__ void normalize(cuDoubleComplex* input, int* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        double realPart = cuCreal(input[idx]);
        int roundedResult = static_cast<int>(round(realPart / n));
        output[idx] = roundedResult;
    }
}

__global__ void mergeDft(cuDoubleComplex* res, int n, bool inverse) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    auto even = res[idx]; // odd = res[idx+n/2]; 
    double angle = -2.0 * M_PI * idx / n;
    cuDoubleComplex omega = make_cuDoubleComplex(cos(angle), sin(angle));
    if (inverse) omega = cuConj(omega);
    res[idx] = cuCadd(even, cuCmul(omega, res[idx+n/2]));
    res[idx+n/2] = cuCsub(even, cuCmul(omega, res[idx+n/2]));
}

__host__ void fastDftRecursive(cuDoubleComplex* arr, cuDoubleComplex* res, int n, bool inverse) {
    if (n <= 1) {
        // res[0] = arr[0]; 
        // Above operation no longer necessary due to cublasZcopy(handle, N, arr, 1, res, 1);
        return;
    }

    fastDftRecursive(arr, res, n/2, inverse);
    fastDftRecursive(arr+n/2, res+n/2, n/2, inverse);

    int blockSize = min(n/2, 256), numBlocks = (n/2 + blockSize - 1) / blockSize;
    mergeDft<<<numBlocks, blockSize>>> (res, n, inverse);
    cudaDeviceSynchronize();
}

__host__ std::string intVecToString(const vector<int> &v) {
    std::ostringstream oss;
    bool leadingZero = true;
    for (const int x : v) {
        if (x == 0 && leadingZero) continue;
        leadingZero = false;
        oss << x;
    } 
    return oss.str();
}

__host__ bool isNumber(const std::string& str) {
    return std::accumulate(str.begin(), str.end(), true, [](bool result, char c) {
        return result && std::isdigit(c);
    });
}

// This function can also be on GPU, but implementation details might be trickier and more nuanced
__host__ void carryOver(vector<int> &num) {
    for (int i = 0; i < num.size()-1; ++i) {
        if (num[i] >= 0) {
            num[i+1] += num[i] / 10; 
            num[i] %= 10;
        }
    }
}

__host__ std::string multiply_large_number_polynomials(const std::string &num1_str = "1234", 
        const std::string &num2_str = "98765") {
    cublasHandle_t handle; cublasCreate(&handle);
    int N = 2;
    while (N < num1_str.length()+num2_str.length()) N *= 2;
    int blockSize = min(N, 256), numBlocks = (N + blockSize - 1) / blockSize;
    vector<cuDoubleComplex> num1, num2; num1.reserve(N); num2.reserve(N);
    // Can perfectly forward (do without make_cuDoubleComplex)? Below chose the safer approach
    for (auto it = num1_str.rbegin(); it != num1_str.rend(); ++it) num1.emplace_back(make_cuDoubleComplex(static_cast<double>((*it)-'0'), 0.0));
    for (auto it = num2_str.rbegin(); it != num2_str.rend(); ++it) num2.emplace_back(make_cuDoubleComplex(static_cast<double>((*it)-'0'), 0.0));
    num1.resize(N); num2.resize(N);

    // Memory needed
    cuDoubleComplex *g_num1_arr, *g_num1_shuffled, *g_num1_res;
    cuDoubleComplex *g_num2_arr, *g_num2_shuffled, *g_num2_res, *g_result_num;
    vector<std::reference_wrapper<cuDoubleComplex*>> v_m {std::ref(g_num1_arr), std::ref(g_num1_shuffled), std::ref(g_num1_res), 
            std::ref(g_num2_arr), std::ref(g_num2_shuffled), std::ref(g_num2_res), std::ref(g_result_num)};
    for (auto &ptr : v_m) cudaMalloc(&(ptr.get()), N * sizeof(cuDoubleComplex));
    [[maybe_unused]] auto cudaFreeDeleter = [](cuDoubleComplex* ptr) { cudaFree(ptr); }; 
    // Uncanonical and "bad" use of unique_ptr; would better use make_unique. This is just to make sure the memory gets deallocated even when exceptions happen
    /* vector<CUDAUniquePtr<cuDoubleComplex, decltype(cudaFreeDeleter)>> v_m {CUDAUniquePtr(g_num1_arr, cudaFreeDeleter), CUDAUniquePtr(g_num1_shuffled),
         CUDAUniquePtr(g_num1_res), CUDAUniquePtr(g_num2_arr), CUDAUniquePtr(g_num2_shuffled), CUDAUniquePtr(g_num2_res), CUDAUniquePtr(g_result_num)};
    */

    // Copy the two complex vectors from host to GPU
    cudaMemcpy(g_num1_arr, num1.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(g_num2_arr, num2.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // Shuffling and pre-copying; absolutely necessary. See what shuffle does in more detail in the notebook
    // Pre-shuffling before the main operation was intended to minimize frequent and chaotic element movements, improving memory locality and coalescing
    shuffle<<<numBlocks, blockSize>>>(g_num1_arr, g_num1_shuffled, N);
    shuffle<<<numBlocks, blockSize>>>(g_num2_arr, g_num2_shuffled, N);
    cublasZcopy(handle, N, g_num1_shuffled, 1, g_num1_res, 1);
    cublasZcopy(handle, N, g_num2_shuffled, 1, g_num2_res, 1);

    // Forward FFT for the two large number polynomials
    fastDftRecursive(g_num1_shuffled, g_num1_res, N, false);
    fastDftRecursive(g_num2_shuffled, g_num2_res, N, false);
    DEBUG_LOG(g_num1_res, N)
    DEBUG_LOG(g_num2_res, N)

    // Compute the product of the 2 Fourier transformations
    hadamargProduct<<<numBlocks, blockSize>>>(g_num1_res, g_num2_res, g_num1_res, N);
    shuffle<<<numBlocks, blockSize>>>(g_num1_res, g_num1_shuffled, N);
    cublasZcopy(handle, N, g_num1_shuffled, 1, g_result_num, 1);

    // Inverse transform
    fastDftRecursive(g_num1_shuffled, g_result_num, N, true);
    DEBUG_LOG(g_result_num, N)

    // Normalize, round, and take real parts (imaginary parts should all be 0) 
    int *g_result_num_int; cudaMalloc(&g_result_num_int, N * sizeof(int));
    normalize<<<numBlocks, blockSize>>>(g_result_num, g_result_num_int, N);

    // Copy the final result back to host 
    vector<int> result_num_int(N);
    cudaMemcpy(result_num_int.data(), g_result_num_int, N * sizeof(int), cudaMemcpyDeviceToHost);
    carryOver(result_num_int); std::reverse(result_num_int.begin(), result_num_int.end());

    // Cleanup
    cudaFree(g_result_num_int);
    for (cuDoubleComplex *ptr : v_m) cudaFree(ptr);

    // Final Answer
    return intVecToString(result_num_int);
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <(large)number1> <(large)number2>" << std::endl;
        return 1;
    }

    std::string num1_str(argv[1]), num2_str(argv[2]);

    if (!isNumber(num1_str) || !isNumber(num2_str)) {
        std::cerr << "Both arguments must be numbers containing only digits!" << std::endl;
        return 1;
    }
    
    std::string final_ans(multiply_large_number_polynomials(num1_str, num2_str)); 
    std::cout << "Product of " << num1_str << " and " << num2_str <<  " is " << final_ans << std::endl;
    return 0;
}