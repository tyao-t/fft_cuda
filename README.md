# CUDA Fast Fourier Transform for Large Integer Multiplication

This repository provides an implementation of the fast Fourier transform (FFT) using CUDA C++ to multiply two very large polynomials with a time complexity of $O(N \ \log \ N)$. The primary application of this project is the efficient multiplication of large integers (on the order of magnitude $10^5$ and higher).

## Overview

This project is based on the Cooley-Tukey FFT algorithm, one of the most well-known and straightforward algorithms to implement for computing the FFT. While the implementation here is not as optimized or advanced as NVIDIA’s cuFFT library, it is a fully viable and working version for educational purposes and practical use. The implementation leverages CUDA functionalities such as `cuComplex` and `cuBLAS` to perform the necessary computations on a GPU.

## Features

- **CUDA Implementation**: The core FFT algorithm is implemented in CUDA C++ for execution on NVIDIA GPUs.
- **Efficient Multiplication**: The application can multiply very large integers in $O(N \ \log \ N)$ time using FFT.
- **Validation**: The implementation includes validation scripts to verify correctness using randomly generated and concatenated test cases.
- **Optimization**: Optimizations including memory coalescing, thread coarsening, and the use of shared memory, were considered and applied where possible.

## Project Structure

- `src/fft.cu`: The main CUDA C++ implementation of the FFT algorithm.
- `src/fft.ipynb`: Jupyter notebook (Google Colab) that demonstrates the CUDA C++ implementation and validates its correctness.
- `src/secondary/fft.py`: Python implementation of the FFT algorithm.
- `bin/fft`: Compiled binary for running the FFT multiplication (generated after compilation; not included by git).

## Dependencies

- **CUDA Toolkit**: Required for compiling and running the CUDA C++ code.
- **cuBLAS**: Used for vector operations on the GPU.
- **cuComplex**: Used for complex number operations.
- **shuf**: Utilized for generating random data for validation.
- **bc**: Used for arbitrary precision arithmetic during validation.

## Getting Started

### Prerequisites

- An NVIDIA GPU with CUDA support.
- CUDA Toolkit installed on your system.

### Compilation

You can compile the CUDA source file using the following command:

```bash
mkdir -p bin && \
nvcc -O3 src/fft.cu -lcublas -UENABLE_DEBUG -o bin/fft
```

### Running the Program

Once compiled, you can run the program with two large integers as input:

```bash
./bin/fft ${num1} ${num2}
```

For example:

```bash
./bin/fft 3141592653589793238462643383279502884197169399375105820974944592 2718281828459045235360287471352662497757247093699959574966967627
```

### Running on Google Colab

If your local computer does not have an NVIDIA GPU, you can run the code on Google Colab, which provides free access to GPU resources. The Jupyter notebook `src/fft.ipynb` is designed to be executed on Google Colab, and it walks through the entire process, including validation steps.

### Validation

To validate the correctness of the FFT implementation, you can run the provided script:

```bash
./run_validate_fft
```

Alternatively, you can directly explore the Jupyter notebook `src/fft.ipynb`, which explains the implementation and validates its correctness.

## Limitations

- **Performance**: This implementation is not as optimized as cuFFT and is intended primarily for educational purposes.
- **Hardware Dependency**: Requires an NVIDIA GPU with CUDA support for execution. Alternatively, you can run the code on Google Colab if a local GPU is not available.
- **CUDAUniquePtr Status**: The CUDAUniquePtr.h file located in the include/ directory provides a utility for managing CUDA resources, aiming to ensure a fundamental level of exception safety by preventing resource leaks. While this header is a viable and functional utility for managing CUDA resources, it is still considered experimental and has not undergone rigorous testing. As a result, it has not been fully integrated into the main project and is deprecated for the time being.

## References

This project was inspired by the [Codeforces Blog](https://codeforces.com/blog/entry/111371), where the theory and algorithm for FFT was discussed.
