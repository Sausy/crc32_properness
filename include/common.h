#pragma once
#include <vector>
#include <cstdint>
#include <iomanip>
#include <iostream>

// Incremental Vector like a bit vector
void incVec(std::vector<uint8_t> &data, unsigned int increment = 1);

// Print a vector
void printVector(const std::vector<uint8_t> &v);

// Generate a lookup table for the number of 1s in each possible byte value (0-255)
std::vector<uint8_t> generateLookupTable();

// XORs two vectors element-wise and returns the result
std::vector<uint8_t> xorVectors(std::vector<uint8_t> &v1, std::vector<uint8_t> &v2);

/**
 * Concatenates two vectors into a single vector.
 *
 * This function template accepts vectors of any type T and returns a new vector
 * that is the result of appending all elements of the second vector to the first.
 *
 * @param vec1 The first vector.
 * @param vec2 The second vector.
 * @return A new vector containing all elements of vec1 followed by all elements of vec2.
 */
template <typename T>
std::vector<T> concatenateVectors(const std::vector<T> &vec1, const std::vector<T> &vec2)
{
    std::vector<T> result;
    result.reserve(vec1.size() + vec2.size());             // Reserve space to improve efficiency
    result.insert(result.end(), vec1.begin(), vec1.end()); // Append all elements of vec1
    result.insert(result.end(), vec2.begin(), vec2.end()); // Append all elements of vec2
    return result;
}


// Helper function to print matrices for debugging purposes
template <typename T>
void printMatrix(const std::vector<std::vector<T>> &matrix)
{
    for (const auto &row : matrix)
    {
        for (auto val : row)
        {
            std::cout << (int)val << " ";
        }
        std::cout << "\n";
    }
}
