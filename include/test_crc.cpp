#include "CRC.h"
#include <iostream>
#include <cassert>

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

int main()
{
    // Testing CRC32 with polynomial 0xF1922815
    std::cout << "====================" << "\nStarting CRC32 Test" << "\n====================" << std::endl;
    std::cout << "Polynomial: 0xF1922815" << std::endl;
    std::vector<uint32_t> lut_crc32 = CRC::createLUT(0xF1922815, 32, false);
    std::cout << "LUT Size: " << lut_crc32.size() << std::endl; // Should be "256
    assert(lut_crc32.size() == 256);
    std::vector<uint8_t> data_crc32 = {0x01, 0x02, 0x03, 0x04}; // Example data
    uint32_t crc32 = CRC::computeCRC(lut_crc32, data_crc32, 32, 0xFFFFFFFF, true, true, true);
    std::cout << "CRC32 Result: " << std::hex << crc32 << std::endl;
    assert(crc32 == 0xB12A406E); // Expected CRC32 value for the example data
    printLUT(lut_crc32);

    // Testing CRC16 with polynomial 0x8005
    std::cout << "\n====================" << "\nStarting CRC16 Test" << "\n====================" << std::endl;
    std::cout << "Polynomial: 0x8005" << std::endl;
    std::vector<uint32_t> lut_crc16 = CRC::createLUT(0x8005, 16, false);
    std::cout << "LUT Size: " << lut_crc16.size() << std::endl; // Should be "256
    assert(lut_crc16.size() == 256);
    std::vector<uint8_t> data_crc16 = {0x01, 0x02, 0x03, 0x04}; // Example data
    uint32_t crc16 = CRC::computeCRC(lut_crc16, data_crc16, 16, 0xFFFF, true, true, true);
    std::cout << "CRC16 Result: " << std::hex << crc16 << std::endl;
    assert(crc16 == 0xD45E); // Expected CRC32 value for the example data
    printLUT(lut_crc16);

    // Testing Generator Matrix for CRC32
    auto gen_matrix_crc32 = CRC::generatorMatrix(0xF1922815, 26); // k = 26 for CRC32
    std::cout << "Generator Matrix for CRC32:\n";
    printMatrix(gen_matrix_crc32);

    // Testing Parity Check Matrix for CRC16
    auto parity_check_matrix_crc16 = CRC::generateParityCheckMatrix(0x8005, 10); // k = 10 for CRC16
    std::cout << "Parity Check Matrix for CRC16:\n";
    printMatrix(parity_check_matrix_crc16);

    return 0;
}