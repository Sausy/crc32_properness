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
#include "tests.cuh"
#include <thrust/device_vector.h>

namespace tests
{
    /*
    * FUNCTION: std::vector<std::vector<unsigned char>> generateBuffers
    *
    * ARGS:
    * size_t count - Number of input buffers to generate.
    * size_t size - Size of each input buffer to generate.
    *
    * DESCRIPTION:
    * This function generates a vector of input buffers with random data.
    * It takes two arguments - 'count' and 'size' - to specify the number of buffers to generate and the size of each buffer.
    * The function uses a random number generator to fill each buffer with unsigned char values ranging from 0 to 255.
    * The generated buffers are stored in a vector of vectors of unsigned char.
    *
    * RETURN VALUE:
    * std::vector<std::vector<uint8_t>> - A vector of input buffers with random data.
    * This function returns the generated buffers as a vector of vectors of unsigned char.
    */
    std::vector<std::vector<uint8_t>> generateBuffers(size_t count, size_t size)
    {
        std::vector<std::vector<uint8_t>> buffers(count, std::vector<uint8_t>(size));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 255);
        for (size_t i = 0; i < count; ++i)
            for (size_t j = 0; j < size; ++j)
                buffers[i][j] = static_cast<uint8_t>(dis(gen));
        return buffers;
    }

    /*
    * FUNCTION: std::vectorstd::string sha512BuffersGPU
    *
    * ARGS:
    * const std::vector<std::vector<uint8_t>>& buffers - A vector of input buffers to calculate SHA-512 hashes.
    *
    * DESCRIPTION:
    * This function calculates the SHA-512 hash for a vector of input buffers on the GPU using CUDA parallel processing.
    * It allocates GPU memory for input and output buffers, copies input buffers from host to GPU memory, and launches a CUDA kernel function to perform the hash calculation.
    * The results are then copied back from GPU to host memory using CUDA streams for faster copying.
    * Finally, the function converts the hash results from binary to hexadecimal string format and returns them as a vector of strings.
    *
    * RETURN VALUE: std::vector<std::vector<uint8_t>>
    * A vector of SHA-512 hashes for the input buffers.
    */
    std::vector<std::vector<uint8_t>> testSHA512GPU(const std::vector<std::vector<uint8_t>>& buffers)
    {
        int numInputs = buffers.size();
        /* Size of each input buffer (assuming all strings have the same size) */
        size_t bufferSize = buffers[0].size();
        int bufferLength = static_cast<int>(bufferSize);

        /*  Create and copy input buffers to GPU memory */
        char* d_inputs;
        cudaMalloc((void**)&d_inputs, numInputs * bufferLength);
        for (int i = 0; i < numInputs; ++i)
            cudaMemcpy(d_inputs + i * bufferLength, buffers[i].data(), bufferLength, cudaMemcpyHostToDevice);

        unsigned char* d_outputs;
        /* 128 - size of SHA-512 hash in bytes */
        cudaMalloc((void**)&d_outputs, numInputs * hashes::DIGEST_SIZE);

        /* Calculate grid size and block size for CUDA threads */
        const int blockSize = 256;
        const int gridSize = (numInputs + blockSize - 1) / blockSize;

        /* Call the sha512Kernel CUDA kernel function on GPU to calculate hashes for each input buffer and save results into output buffer */
        hashes::sha512Kernel << <gridSize, blockSize >> > (d_inputs, numInputs, d_outputs, bufferLength);

        /* Allocate memory on host for results */
        std::vector<std::vector<uint8_t>> results(numInputs);
        /* Allocate memory on host for output buffer */
        std::vector<unsigned char> h_outputs(numInputs * hashes::DIGEST_SIZE);

        /* Create CUDA stream for faster copying */
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        /* Copy results using CUDA stream */
        cudaMemcpyAsync(h_outputs.data(), d_outputs, numInputs * hashes::DIGEST_SIZE, cudaMemcpyDeviceToHost, stream);
        /* Synchronize CUDA stream to complete copying */
        cudaStreamSynchronize(stream);

        /* Copy results to vector of vectors */
        for (int i = 0; i < numInputs; ++i) {
            results[i].resize(hashes::DIGEST_SIZE);
            memcpy(results[i].data(), h_outputs.data() + i * hashes::DIGEST_SIZE, hashes::DIGEST_SIZE);
        }
        /* Free GPU memory */
        cudaFree(d_inputs);
        cudaFree(d_outputs);
        /* Destroy CUDA stream */
        cudaStreamDestroy(stream);
        return results;
    }

    /*
    * FUNCTION: std::vector<std::string> sha512BuffersCPU
    *
    * ARGS:
    * std::vector<std::vector<uint8_t>> const& buffers - A vector of input buffers to calculate SHA-512 hashes.
    *
    * DESCRIPTION:
    * This function calculates the SHA-512 hash for all the strings in the vector 'buffers' on the CPU using multiple threads for parallel execution.
    * It launches multiple threads to calculate SHA-512 hash for each string in the vector 'buffers' concurrently using std::async and std::future.
    * Then it waits for all the threads to complete and retrieves the results.
    *
    * RETURN VALUE: std::vector<std::vector<uint8_t>>
    * A vector of SHA-512 hashes for the input buffers.
    */
    std::vector<std::vector<uint8_t>> testSHA512CPU(std::vector<std::vector<uint8_t>> const& buffers)
    {
        const int numBuffers = buffers.size();
        /* Launch multiple threads for calculating SHA-512 hash concurrently using std::async and std::future */
        std::vector<std::future<std::vector<uint8_t>>> futures;
        for (int i = 0; i < numBuffers; ++i)
            futures.emplace_back(std::async(std::launch::async, [](std::vector<uint8_t> const& buffer) -> std::vector<uint8_t> {
            return hashes::sha512(buffer.data(), buffer.size());
                }, buffers[i]));

        /* Wait for all threads to complete and retrieve results */
        std::vector<std::vector<uint8_t>> hashResults;
        for (auto& future : futures)
            hashResults.push_back(future.get());

        /* Return a vector containing the calculated SHA-512 hashes for all the buffers in the 'buffers' vector */
        return hashResults;
    }

    /*
    * FUNCTION: std::vector<uint32_t> testCRC32CPU
    *
    * ARGS:
    * std::vector<std::vector<uint8_t>> const& buffers - A vector of input buffers to calculate CRC32 checksums.
    *
    * DESCRIPTION:
    * This function calculates the CRC32 checksum for buffers of random data on the GPU using CUDA.
    * It dynamically àllocates memory on the device (GPU) for the buffers and CRC32 results.
    * The function launches a CUDA kernel on the device to calculate the CRC32 checksum for each buffer in parallel.
    * It then copies the results back from the device to the host and frees the allocated memory.
    *
    * RETURN VALUE: std::vector<uint32_t>
    * CRC32 checksums for each buffer in the input vector on the GPU.
    */
    std::vector<uint32_t> testCRC32GPU(std::vector<std::vector<uint8_t>> const& buffers)
    {
        const int numBuffers = buffers.size();
        const int bufferSize = buffers[0].size();

        cudaError_t cudaStatus;

        /* Dynamic memory allocation on the device */
        unsigned char* d_buffers;
        uint32_t* d_crcResults;
        cudaStatus = cudaMalloc(reinterpret_cast<void**>(&d_buffers), numBuffers * bufferSize * sizeof(unsigned char));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            std::cout << "cudaMalloc failed!"; 
        }
        cudaStatus = cudaMalloc(reinterpret_cast<void**>(&d_crcResults), numBuffers * sizeof(uint32_t));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            std::cout << "cudaMalloc failed!";
        }

        /* Copy data from host to device using cudaMemcpy2D */
        for (int i = 0; i < numBuffers; ++i)
            cudaMemcpy(d_buffers + i * bufferSize, buffers[i].data(), bufferSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

        /* Calculate number of blocks and threads per block for the kernel launch */
        const int blockSize = 256;
        const int numBlocks = (numBuffers + blockSize - 1) / blockSize;

        /* Launch the kernel on the device indicating the number of blocks(numBlocks) and block size(blockSize) that will be used for parallel execution of calculations on the GPU. */
        crc32::crc32Kernel << <numBlocks, blockSize >> > (d_buffers, d_crcResults, numBuffers, bufferSize);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            std::cout << "addKernel launch failed";
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            std::cout << "cudaDeviceSynchronize returned error code %d after launching addKernel!\n";
        }

        /* Copy results from device to host directly into a vector without intermediate buffer */
        std::vector<uint32_t> checksums(numBuffers);
        cudaStatus = cudaMemcpy(checksums.data(), d_crcResults, numBuffers * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
        }

        /* Free device memory */
        cudaFree(d_buffers);
        cudaFree(d_crcResults);
        /* Return the CRC32 checksums as a vector */
        return checksums;
    }

    /*
    * FUNCTION: std::vector<uint32_t> testCRC32CPU
    *
    * ARGS:
    * std::vector<std::vector<uint8_t>> const& buffers - Vector of buffers to calculate the CRC32 checksum for.
    *
    * DESCRIPTION:
    * This function calculates the CRC32 checksum for a vector of vectors of unsigned char buffers on the CPU using multiple threads for parallel execution.
    * It launches multiple threads to calculate the CRC32 checksum for each buffer in parallel using std::async and std::future.
    * It then waits for all threads to complete and retrieves the results.
    *
    * RETURN VALUE: std::vector<uint32_t>
    * A vector of CRC32 checksums for the input buffers.
    */
    std::vector<uint32_t> testCRC32CPU(std::vector<std::vector<uint8_t>> const& buffers)
    {
        const int numBuffers = buffers.size();
        /* Launch multiple threads for CRC32 calculation in parallel using std::async and std::future */
        std::vector<std::future<uint32_t>> futures;
        for (int i = 0; i < numBuffers; i++)
            futures.emplace_back(std::async(std::launch::async, [](std::vector<uint8_t> const& buffer) -> uint32_t {
            return crc32::crc32(buffer.data(), buffer.size());
                }, buffers[i]));

        /* Wait for all futures to complete and get the results */
        std::vector<uint32_t> crcResults;
        for (auto& future : futures)
            crcResults.push_back(future.get());

        /* Return the time taken to calculate the CRC32 checksums for all buffers on the CPU */
        return crcResults;
    }
}