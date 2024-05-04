#include "CRC.h"
#include <iostream>
#include <cassert>
#include "common.h"


std::vector<double> polyDivide(const std::vector<double>& dividend, const std::vector<double>& divisor, std::vector<double>& remainder) {
    int n = dividend.size();
    int m = divisor.size();
    std::vector<double> quotient(n - m + 1, 0);
    
    // Copy dividend to remainder initially
    remainder = dividend;

    // Division Algorithm
    for (int i = n - m; i >= 0; i--) {
        quotient[i] = remainder[m + i - 1] / divisor[m - 1];
        for (int j = m + i - 1; j >= i; j--) {
            remainder[j] -= quotient[i] * divisor[j - i];
        }
    }

    // Trim leading zeros in remainder
    while (!remainder.empty() && remainder.back() == 0) {
        remainder.pop_back();
    }

    return quotient;
}

// TODO: Thsi function is actually in common.h defined i dont know why it 
// cannot find it 
// workaround is to copy the function here
std::vector<uint8_t> BitShiftVector(const std::vector<uint8_t> &vec, int shift) {
    // shift value to the left 
    // if bit is shifted over the 8bit boundary it is added to the next byte
    std::vector<uint8_t> shifted_vec(vec.size(), 0);
    for (int i = 0; i < vec.size(); i++) {
        shifted_vec[i] = vec[i] << shift;
        if (i < vec.size() - 1) {
            shifted_vec[i] |= vec[i + 1] >> (8 - shift);
        }
    }
    return shifted_vec;
}

int main()
{
    
    //std::vector<uint8_t> message = {0x00, 0x00, 0x00, 0x01}; // Example message
    std::vector<uint8_t> message = {0x00, 0x00, 0x00, 0x88 }; // Example message
    uint64_t polynomial = 0xB;  // Example polynomial (CRC-32)
    int r = 3;  // CRC bit length for CRC-32
    int k = 4;

    uint64_t crc = CRC::calculateCRC_unitVec(message, polynomial, r, k);
    std::cout << "CRC value: " << std::hex << crc << std::endl;


    std::cout << "message: ";
    for (auto m : message) {
        std::cout << std::hex << (int)m << " ";
    }
    
    std::vector<uint8_t> message_shifted = BitShiftVector(message, 1);
    // print the shifted message
    std::cout << "Shifted message: ";
    for (auto m : message_shifted) {
        std::cout << std::hex << (int)m << " ";
    }
    

    /*
    //std::vector<double> dividend = {0, 0, 0, 1};  // x^3
    //std::vector<double> divisor = {1, 1, 0, 1};  // x^3 + x + 1

    //std::vector<double> dividend = {0, 0, 0, 0, 1}; // x^4
    //std::vector<double> divisor = {1, 1, 0, 1};  // x^3 + x + 1

    std::vector<double> dividend = {0, 0, 0, 0, 0, 1}; // x^5
    std::vector<double> divisor = {1, 1, 0, 1};  // x^3 + x + 1

    std::vector<double> remainder;

    std::vector<double> quotient = polyDivide(dividend, divisor, remainder);

    std::cout << "Quotient: ";
    for (auto q : quotient) {
        std::cout << q << " ";
    }
    std::cout << std::endl << "Remainder: ";
    for (auto r : remainder) {
        std::cout << r << " ";
    }
    std::cout << std::endl;

    return 0;
    */

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


    //====================================
    //====== SYSTEMATIC GENERATOR MATRIX ======
    // Check if polynom is too long
    std::cout << "\n===================="
              << "\nStarting Systematic Generator Matrix Test"
              << "Generator Matrix r=3, k=4\n"
              << "poly: 0xFB for CRC4 Hamming(7,4)\n"
              << "counter check with [Error Control Coding by Shu Lin] page 55"
              << "\n====================" << std::endl;
    syst_gen_matrix_7_4_hamming = CRC::SystematicGeneratorMatrix(0xB, 3, 4);
    gen_matrix_7_4_hamming = CRC::generatorMatrix(0xFB, 3, 4);
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
    assert(syst_gen_matrix_7_4_hamming == G_correct);

    return 0;
}