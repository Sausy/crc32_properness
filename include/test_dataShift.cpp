#include <iostream>
#include <vector>
#include <cstdint>

uint64_t foo(const std::vector<uint8_t> &data, int k, int shift)
{
    // Start with a uint64_t to accumulate the bits
    uint64_t result = 0;

    // Load bits from vector to the result considering the vector as big-endian
    for (size_t i = 0; i < data.size(); ++i) {
        // Shift left by 8 each time as we add a new byte, then OR with the new byte
        result = (result << 8) | data[i];
    }

    // Adjust result based on bit length k
    // Remove the bits above k by shifting left to remove excess bits and then shifting back right
    if (k < 64) {
        result = (result << (64 - k)) >> (64 - k);
    }

    // Apply the shift: align the desired bits to the least significant bits of the result
    if (shift > 0) {
        // Calculate how many bits to shift right to get the desired output
        int shift_amount = k - shift;
        result = (result >> shift_amount) & ((1ULL << shift) - 1);
    } else {
        result = 0; // If shift is zero, return zero
    }

    return result;
}

int main()
{
    // Example 1
    std::vector<uint8_t> data1 = {0x40, 0x03};
    int k1 = 15;
    int shift1 = 1;
    uint64_t result1 = foo(data1, k1, shift1);
    std::cout << "Example 1 Result: 0x" << std::hex << result1 << std::endl;

    // Example 2
    std::vector<uint8_t> data2 = {0x40, 0x03};
    int k2 = 15;
    int shift2 = k2;
    uint64_t result2 = foo(data2, k2, shift2);
    std::cout << "Example 2 Result: 0x" << std::hex << result2 << std::endl;

    for (auto k = 0; k < 16; ++k){
        result2 = foo(data2, k2, k);
        std::cout << "Example " << k << " Result: 0x" << std::hex << result2 << std::endl;
    }

    // Example 3
    std::vector<uint8_t> data2 = {0x00, 0x00, 0x01};
    int k2 = 3;
    int shift2 = k2;
    uint64_t result2 = foo(data2, k2, shift2);
    std::cout << "Example 2 Result: 0x" << std::hex << result2 << std::endl;

    for (auto k = 0; k < 16; ++k){
        result2 = foo(data2, k2, k);
        std::cout << "Example " << k << " Result: 0x" << std::hex << result2 << std::endl;
    }

    return 0;
}