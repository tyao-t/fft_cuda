#ifndef CUDAUNIQUEPTR_H
#define CUDAUNIQUEPTR_H

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <functional>

template <typename T, typename Deleter = std::function<void(T*)>>
class CUDAUniquePtr {
private:
    T* ptr;
    Deleter deleter;

public:
    explicit CUDAUniquePtr(size_t size=1, Deleter d = [](T* p) { cudaFree(p); })
        : ptr(nullptr), deleter(d) {
        cudaError_t err = cudaMalloc(&ptr, size * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
        }
    }

    ~CUDAUniquePtr() {
        if (ptr) {
            deleter(ptr);
        }
    }

    CUDAUniquePtr(const CUDAUniquePtr&) = delete;
    CUDAUniquePtr& operator=(const CUDAUniquePtr&) = delete;

    CUDAUniquePtr(CUDAUniquePtr&& other) noexcept
        : ptr(other.ptr), deleter(std::move(other.deleter)) {
        other.ptr = nullptr;
    }

    CUDAUniquePtr& operator=(CUDAUniquePtr&& other) noexcept {
        if (this != &other) {
            if (ptr) {
                deleter(ptr);
            }
            ptr = other.ptr;
            deleter = std::move(other.deleter);
            other.ptr = nullptr;
        }
        return *this;
    }

    T* get() const { return ptr; }

    T& operator[](size_t index) { return ptr[index]; }

    T* release() {
        T* tmp = ptr;
        ptr = nullptr;
        return tmp;
    }
};

#endif // CUDAUNIQUEPTR_H
