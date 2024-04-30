/*
 *
 *  Copyright (C) 2023, SToFU Systems S.L.
 *  All rights reserved.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "tests.cuh"

inline uint64_t getTime()
{
    return std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
}
// Function to print all buffers and their byte values in hexadecimal
void printBuffers(const std::vector<std::vector<unsigned char>>& buffers) {
    int bufferIndex = 0;
    for (const auto& buffer : buffers) {
        std::cout << "Buffer " << bufferIndex << " (" << buffer.size() << " bytes): ";
        for (unsigned char byte : buffer) {
            std::cout << std::hex << std::setfill('0') << std::setw(2)
                << static_cast<int>(byte) << " ";
        }
        std::cout << std::endl;
        bufferIndex++;
    }
    // Resetting cout to decimal to avoid affecting subsequent outputs
    std::cout << std::dec;
}


void crc32Ex(size_t count, size_t size)
{
    std::vector<std::vector<unsigned char>> buffers = tests::generateBuffers(count, size);

    auto startGPU = getTime(); // Start time for GPU function
    auto resultGPU = tests::testCRC32GPU(buffers); // Call the GPU function
    std::cout << "GPU crc32 Execution Time: " << getTime() - startGPU << " milliseconds" << std::endl; // Execution time of GPU function in milliseconds

    auto startCPU = getTime(); // Start time for CPU function
    auto resultCPU = tests::testCRC32CPU(buffers); // Call the CPU function
    std::cout << "CPU crc32 Execution Time: " << getTime() - startCPU << " milliseconds" << std::endl; // Execution time of CPU function in milliseconds

    //std::vector<uint32_t> checksums(numBuffers);
    //printBuffers(buffers);

    //std::vector<uint32_t> checksums(numBuffers);
    //for (uint32_t checksum : resultGPU) {
    //    std::cout << "crc32: 0x" << std::hex << std::setfill('0') << std::setw(8) << checksum << std::endl;
    //}
}

void sha512Ex(size_t count, size_t size)
{
    std::vector<std::vector<unsigned char>> buffers = tests::generateBuffers(count, size);

    auto startGPU = getTime(); // Start time for GPU function
    auto resultGPU = tests::testSHA512GPU(buffers); // Call the GPU function
    std::cout << "GPU sha512 Execution Time: " << getTime() - startGPU << " milliseconds" << std::endl; // Execution time of GPU function in milliseconds

    auto startCPU = getTime(); // Start time for CPU function
    auto resultCPU = tests::testSHA512CPU(buffers); // Call the CPU function
    std::cout << "CPU sha512 Execution Time: " << getTime() - startCPU << " milliseconds" << std::endl; // Execution time of CPU function in milliseconds
}

void cuda_device_info() {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
}

int main()
{
    cuda_device_info();

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        std::cout << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?";
        //return -1;
    }

    crc32Ex(10000, 4);
    //sha512Ex(10, 32);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}