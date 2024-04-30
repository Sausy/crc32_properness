#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <sstream>
#include <stdio.h>
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <future>
#include <thread>

#include "sha512.cuh"
#include "crc32.cuh"

namespace tests
{
    std::vector<std::vector<uint8_t>>       generateBuffers(size_t count, size_t size);

    std::vector<std::vector<uint8_t>>       testSHA512GPU(std::vector<std::vector<uint8_t>> const& buffers);
    std::vector<std::vector<uint8_t>>       testSHA512CPU(std::vector<std::vector<uint8_t>> const& buffers);

    std::vector<uint32_t>                   testCRC32GPU(std::vector<std::vector<uint8_t>> const& buffers);
    std::vector<uint32_t>                   testCRC32CPU(std::vector<std::vector<uint8_t>> const& buffers);

}