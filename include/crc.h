#pragma once
#include <vector>
#include <cstdint>
#include <iomanip>

class CRC
{
public:
    // Creates a lookup table for CRC calculations.
    static std::vector<uint32_t> createLUT(uint32_t polynomial, int type, bool reflected = false);

    // Computes the CRC for a given dataset using a provided lookup table.
    static uint32_t computeCRC(const std::vector<uint32_t> &lut, const std::vector<uint8_t> &data,
                               int type = 32, //defines if 8bit, 16bit or 32bit CRC
                               uint32_t initial = 0xFFFFFFFF, bool inputReflected = false,
                               bool resultReflected = false, bool finalXOR = true);

    // Generates the generator matrix for CRC codes.
    static std::vector<std::vector<uint8_t>> generatorMatrix(uint32_t polynomial, int k);

    // Generates the parity check matrix for CRC codes.
    static std::vector<std::vector<uint8_t>> generateParityCheckMatrix(uint32_t polynomial, int k);

private:
    // Helper function to reflect bits of a number.
    static uint32_t reflect(uint32_t data, int nBits);
};


// Helper function to print lookup tables for debugging purposes
void printLUT(const std::vector<uint32_t> &lut, int columns = 4);