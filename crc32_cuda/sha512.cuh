#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <sstream>
#include <stdio.h>
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>

namespace hashes
{
    static constexpr unsigned int DIGEST_SIZE = 64;
    static constexpr unsigned int HASH_SIZE = 128;

    struct sha512_context {
        uint64_t  length, state[8];
        size_t curlen;
        unsigned char buf[128];
    };

    int         __device__ __host__ sha512_init(sha512_context* c);
    int         __device__ __host__ sha512_final(sha512_context* md, uint8_t* out);
    int         __device__ __host__ sha512_update(sha512_context* md, const uint8_t* in, size_t inlen);
    int         __device__ __host__ sha512(const uint8_t* message, size_t length, uint8_t* out);
    std::vector<uint8_t>   __host__ sha512(const uint8_t* message, size_t length);

    void                 __global__ sha512Kernel(char* inputs, int numInputs, uint8_t* outputs, int bufferLength);
}