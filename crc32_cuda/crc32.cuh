#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>
#include <chrono>

namespace crc32
{
    __device__ __host__ uint32_t reflect(uint32_t val, int num_bits); 
    __device__ __host__ uint32_t crc32(const uint8_t* buf, int size);
    __global__          void     crc32Kernel(_In_ const uint8_t* buffers, _Out_ uint32_t* crcResults, int numBuffers, int bufferSize);
}