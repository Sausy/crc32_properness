#pragma once
#include <vector>
#include <cstdint>
#include <iomanip>

class CRC
{
public:
    // constructor
    CRC(int type = 32, uint32_t polynomial = 0xF1922815, uint32_t initial = 0xFFFFFFFF, bool reflected = true, bool resultReflected = true, bool finalXOR = true);

    // defining parameters
    uint32_t polynomial;       // Polynomial used for CRC calculations
    int type;                  // Type of CRC (8, 16, 32)
    bool reflected;            // Whether the input data is reflected
    bool resultReflected;      // Whether the result is reflected
    bool finalXOR;             // Whether to XOR the final result with 0xFFFFFFFF
    uint32_t initial;          // Initial value for CRC calculations

    std::vector<uint32_t> LUT; // Lookup table for CRC calculations
    std::vector<std::vector<uint8_t>> G; // Generator matrix for CRC codes
    std::vector<std::vector<uint8_t>> systematicG; // Systematic Generator matrix for CRC codes
    std::vector<std::vector<uint8_t>> H; // Parity check matrix for CRC codes

    // Creates a lookup table for CRC calculations.
    static std::vector<uint32_t> createLUT(uint32_t polynomial, int type, bool reflected = false);

    // Computes the CRC for a given dataset using a provided lookup table.
    static uint32_t computeCRC(const std::vector<uint32_t> &lut, const std::vector<uint8_t> &data,
                               int type = 32, //defines if 8bit, 16bit or 32bit CRC
                               uint32_t initial = 0xFFFFFFFF, bool inputReflected = false,
                               bool resultReflected = false, bool finalXOR = true);

    // Generates the generator matrix for CRC codes.
    static std::vector<std::vector<uint8_t>> generatorMatrix(uint32_t polynomial, int r, int k);
    static std::vector<std::vector<uint8_t>> SystematicGeneratorMatrix(uint32_t polynomial, int r, int k);

    // Generates the parity check matrix for CRC codes.
    static std::vector<std::vector<uint8_t>> generateParityCheckMatrix(uint32_t polynomial, int r, int k);

private:
    // Helper function to reflect bits of a number.
    static uint32_t reflect(uint32_t data, int nBits);
};


// Helper function to print lookup tables for debugging purposes
void printLUT(const std::vector<uint32_t> &lut, int columns = 4);