#include "CRC.h"
#include <iostream>
#include <cassert>
#include "common.h"



int main()
{
    // Testing CRC32 with polynomial 0xF1922815
    std::cout << "====================" << "\nStarting CRC32 Test" << "\n====================" << std::endl;
    std::cout << "Polynomial: 0xF1922815" << std::endl;
    std::vector<uint32_t> lut_crc32 = CRC::createLUT(0xF1922815, 32, false);
    std::cout << "LUT Size: " << std::dec << lut_crc32.size() << std::endl; // Should be "256
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
    std::cout << "LUT Size: " << std::dec << lut_crc16.size() << std::endl; // Should be "256
    assert(lut_crc16.size() == 256);
    std::vector<uint8_t> data_crc16 = {0x01, 0x02, 0x03, 0x04}; // Example data
    uint32_t crc16 = CRC::computeCRC(lut_crc16, data_crc16, 16, 0xFFFF, true, true, true);
    std::cout << "CRC16 Result: " << std::hex << crc16 << std::endl;
    assert(crc16 == 0xD45E); // Expected CRC32 value for the example data
    printLUT(lut_crc16);

    // Testing CRC8 with polynomial 0x31
    std::cout << "\n===================="
              << "\nStarting CRC8 Test"
              << "\n====================" << std::endl;
    std::cout << "Polynomial: 0x31" << std::endl;
    std::vector<uint32_t> lut_crc8 = CRC::createLUT(0x31, 8, false);
    std::cout << "LUT Size: " << std::dec << lut_crc8.size() << std::endl; // Should be "256
    assert(lut_crc8.size() == 256);
    std::vector<uint8_t> data_crc8 = {0x01}; // Example data
    uint32_t crc8 = CRC::computeCRC(lut_crc8, data_crc8, 8, 0xFF, true, true, true);
    std::cout << "CRC8 Result: " << std::hex << crc8 << std::endl;
    assert(crc8 == 0x94); // Expected CRC32 value for the example data
    printLUT(lut_crc8);



    std::cout << "\n===================="
              << "\nStarting CRC4 Test"
              << "Generator Matrix r=4, k=3"
              << "\n====================" << std::endl;
    // Testing Generator Matrix for CRC4
    // The Generator polynom: 0xB for CRC4 is 1011 in binary
    // The Generator-Matrix has the size of [k,n]
    auto gen_matrix_crc4 = CRC::generatorMatrix(0xB, 4, 3); // r = 4 for CRC4 and was set to k=3
    auto gen_matrix_crc4_systematic = CRC::SystematicGeneratorMatrix(0xB, 4, 3);
    std::cout << "Generator Matrix for CRC4:\n";
    printMatrix(gen_matrix_crc4);

    std::cout << "Systematic Generator Matrix for CRC4:\n";
    printMatrix(gen_matrix_crc4_systematic);

    if(gen_matrix_crc4[0] != std::vector<uint8_t>{1, 1, 0, 1, 0, 0, 0})
        std::cout << "Error: Generator Matrix for CRC4 is not correct!" << std::endl;

    //assert(gen_matrix_crc4[0] == std::vector<uint8_t>{1, 1, 0, 1, 0, 0, 0});
    assert(gen_matrix_crc4[0][0] == 1);
    assert(gen_matrix_crc4[0][1] == 1);
    assert(gen_matrix_crc4[0][2] == 0);
    assert(gen_matrix_crc4[0][3] == 1);
    assert(gen_matrix_crc4[0][4] == 0);
    assert(gen_matrix_crc4[0][5] == 0);
    assert(gen_matrix_crc4[0][6] == 0);

    //====== SYSTEMATIC GENERATOR MATRIX ======
    // Test default Hamming(7,4) code meaning k=4 and r= 3
    /*
     * The parity check matrix shall be tested according to
     * [Error Control Coding by Shu Lin]
     * page 52. given message polynomial m(x) = 1 + x^2 + x^3
     * page 55. given the Generator Matrix
     * page 77. calculation of weight distribution of dual "B"
     * page 77. paritiy check matrix H
     */
    std::cout << "\n===================="
              << "\nStarting Systematic Generator Matrix Test"
              << "Generator Matrix r=3, k=4\n"
              << "poly: 0xB for CRC4 Hamming(7,4)\n"
              << "counter check with [Error Control Coding by Shu Lin] page 55"
              << "\n====================" << std::endl;
    auto syst_gen_matrix_7_4_hamming = CRC::SystematicGeneratorMatrix(0xB, 3, 4);
    auto gen_matrix_7_4_hamming = CRC::generatorMatrix(0xB, 3, 4);
    std::cout << "CRC3 G-Generator Matrix\n";
    printMatrix(gen_matrix_7_4_hamming);
    std::cout << "Systematic Generator Matrix Hamming(7,4):\n";
    printMatrix(syst_gen_matrix_7_4_hamming);
    /* correct G
    please not that in the paper [I | P^T] are swapped 
    but this does not make a difference in the final result if 
    kept consistent

    1 0 0 0 1 1 0
    0 1 0 0 0 1 1
    0 0 1 0 1 1 1
    0 0 0 1 1 0 1
    */

    std::vector<std::vector<uint8_t>> G_correct = {
        {1, 0, 0, 0, 1, 1, 0},
        {0, 1, 0, 0, 0, 1, 1},
        {0, 0, 1, 0, 1, 1, 1},
        {0, 0, 0, 1, 1, 0, 1}
    };
    assert(syst_gen_matrix_7_4_hamming == G_correct);



    //====== PARITY CHECK MATRIX ======
    /*
    * The parity check matrix shall be tested according to 
    * [Error Control Coding by Shu Lin]
    * page 52. given message polynomial m(x) = 1 + x^2 + x^3
    * page 55. given the Generator Matrix 
    * page 77. calculation of weight distribution of dual "B"
    * page 77. paritiy check matrix H
    */
    std::cout << "\n===================="
              << "\nStarting Parity Check Matrix Test"
              << "Generator Matrix r=3, k=4\n"
              << "poly: 0xB for CRC4 Hamming(7,4)\n"
              << "counter check with [Error Control Coding by Shu Lin] page 77"
              << "\n====================" << std::endl;
    // Testing Parity Check Matrix for CRC4
    // The polynom: 0xB for CRC4 is 1011 in binary
    // The Parity Check Matrix has the size of [n-k,n]
    auto parity_check_matrix_crc4 = CRC::generateParityCheckMatrix(0xB, 3, 4); 
    std::cout << "Parity Check Matrix for CRC4:\n";
    printMatrix(parity_check_matrix_crc4);
    /* correct H
    please not that in the paper [P | I] are swapped
    but this does not make a difference in the final result if
    kept consistent

    1 0 1 1 1 0 0
    1 1 1 0 0 1 0
    0 1 1 1 0 0 1

    */
   std::vector<std::vector<uint8_t>> H_correct = {
        {1, 0, 1, 1, 1, 0, 0},
        {1, 1, 1, 0, 0, 1, 0},
        {0, 1, 1, 1, 0, 0, 1}
    };
    assert(parity_check_matrix_crc4 == H_correct);

    /*

    // Testing Parity Check Matrix for CRC16
    auto gen_matrix_crc32 = CRC::generatorMatrix(0xF1922815, 32, 26); // r = 32 for CRC32
    std::cout << "Generator Matrix for CRC32:\n";
    printMatrix(gen_matrix_crc32);


    // Testing Parity Check Matrix for CRC16
    auto parity_check_matrix_crc16 = CRC::generateParityCheckMatrix(0x8005, 16, 10); // k = 10 for CRC16
    std::cout << "Parity Check Matrix for CRC16:\n";
    printMatrix(parity_check_matrix_crc16);
    */

    /*
    std::cout << "\n===================="
              << "\nStarting Parity Check Matrix Test"
              << "No Matrix to counter check but \n"
              << "the size of the matrix can be checked"
              << "Generator Matrix r=8, k=16\n"
              << "poly: 0x1D for CRC8\n"
              << "\n====================" << std::endl;
    // H [n-k,n]
    // G [k,n]
    auto H = CRC::generateParityCheckMatrix(0x1D, 8, 16); 
    auto sysG = CRC::SystematicGeneratorMatrix(0x1D, 8, 16);
    auto G = CRC::generatorMatrix(0x1D, 8, 16);
    std::cout << "CRC8 G-Generator Matrix\n";
    printMatrix(G);
    
    std::cout << "CRC8 Systematic Generator Matrix\n";
    printMatrix(sysG);
    
    std::cout << "CRC8 H-Parity Check Matrix\n";
    printMatrix(H);
    */

    std::cout << "\n===================="
              << "\nStarting Parity Check Matrix Test"
              << "No Matrix to counter check but \n"
              << "the size of the matrix can be checked"
              << "Generator Matrix r=6, k=16\n"
              << "poly: 0x38 for CRC6\n"
              << "\n====================" << std::endl;
    // H [n-k,n]
    // G [k,n]
    auto H = CRC::generateParityCheckMatrix(0x38, 6, 16);
    auto sysG = CRC::SystematicGeneratorMatrix(0x38, 6, 16);
    auto G = CRC::generatorMatrix(0x38, 6, 16);
    std::cout << "CRC6 G-Generator Matrix\n";
    printMatrix(G);

    std::cout << "CRC6 Systematic Generator Matrix\n";
    printMatrix(sysG);

    std::cout << "CRC6 H-Parity Check Matrix\n";
    printMatrix(H);

    return 0;
}