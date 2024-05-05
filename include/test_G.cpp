#include "CRC.h"
#include <iostream>
#include <cassert>
#include "common.h"

int main()
{
    uint64_t crc_correct = 0ull;
    uint64_t crc = 0ull;
    auto r = 8; 
    auto k = 8;
    uint64_t polynomial = 0b100011101;

    std::vector<uint8_t> message = {0xC2}; // Example data
    std::cout << "===================="
              << "\nStarting CRC Test"
              << "\nr = " << std::dec << r
              << "\nk = " << std::dec << k
              << "\n===================="
              << "\ninit: " << std::hex << 0x00
              << "\nReflex IN: " << std::dec << false
              << "\nReflex OUT: " << std::dec << false
              << "\nXOR OUT: " << std::hex << 0x00
              << "\nPolynomial: " << std::hex << polynomial
              << std::endl;
    // print message 
    std::cout << "Message: ";
    printVector(message);
    std::cout << "\n====================" << std::endl;

    crc_correct = 0x0F;
    crc = CRC::computeCRC(polynomial, r, k, message, false, false, false, false, 0ull, false);
    std::cout << "CRC Result: " << std::hex << crc << std::endl;
    assert(crc == crc_correct); // Expected CRC value for the example data

    std::cout << "===================="
              << "\nStarting CRC Test"
              << "\nr = " << std::dec << r
              << "\nk = " << std::dec << k
              << "\n===================="
              << "\ninit: " << std::hex << 0x00
              << "\nReflex IN: " << std::dec << false
              << "\nReflex OUT: " << std::dec << false
              << "\nXOR OUT: " << std::hex << 0xFF
              << "\nPolynomial: " << std::hex << polynomial
              << std::endl;
    // print message
    std::cout << "Message: ";
    printVector(message);
    std::cout << "\n====================" << std::endl;

    //set corret value 
    crc_correct = 0xF0;
    crc = CRC::computeCRC(polynomial, r, k, message, false, false, false, true, 0ull, false);
    std::cout << "CRC Result: " << std::hex << crc << std::endl;
    assert(crc == crc_correct); // Expected CRC value for the example data

    std::cout << "===================="
              << "\nStarting CRC Test"
              << "\nr = " << std::dec << r
              << "\nk = " << std::dec << k
              << "\n===================="
              << "\ninit: " << std::hex << 0x00
              << "\nReflex IN: " << std::dec << true
              << "\nReflex OUT: " << std::dec << false
              << "\nXOR OUT: " << std::hex << 0xFF
              << "\nPolynomial: " << std::hex << polynomial
              << std::endl;
    // print message
    std::cout << "Message: ";
    printVector(message);
    std::cout << "\n====================" << std::endl;

    // set corret value
    crc_correct = 0xCB;
    crc = CRC::computeCRC(polynomial, r, k, message, false, true, false, true, 0ull, false);
    std::cout << "CRC Result: " << std::hex << crc << std::endl;
    assert(crc == crc_correct); // Expected CRC value for the example data



    std::cout << "===================="
              << "\nStarting CRC Test"
              << "\nr = " << std::dec << r
              << "\nk = " << std::dec << k
              << "\n===================="
              << "\ninit: " << std::hex << 0x00
              << "\nReflex IN: " << std::dec << true
              << "\nReflex OUT: " << std::dec << true
              << "\nXOR OUT: " << std::hex << 0xFF
              << "\nPolynomial: " << std::hex << polynomial
              << std::endl;
    // print message
    std::cout << "Message: ";
    printVector(message);
    std::cout << "\n====================" << std::endl;

    // set corret value
    crc_correct = 0xD3;
    crc = CRC::computeCRC(polynomial, r, k, message, false, true, true, true, 0ull ,false);
    std::cout << "CRC Result: " << std::hex << crc << std::endl;
    assert(crc == crc_correct); // Expected CRC value for the example data



    std::cout << "===================="
              << "\nStarting CRC Test"
              << "\nr = " << std::dec << r
              << "\nk = " << std::dec << k
              << "\n===================="
              << "\ninit: " << std::hex << 0xFF
              << "\nReflex IN: " << std::dec << true
              << "\nReflex OUT: " << std::dec << true
              << "\nXOR OUT: " << std::hex << 0xFF
              << "\nPolynomial: " << std::hex << polynomial
              << std::endl;
    // print message
    std::cout << "Message: ";
    printVector(message);
    std::cout << "\n====================" << std::endl;

    // set corret value
    crc_correct = 0xF0;
    crc = CRC::computeCRC(polynomial, r, k, message, false, true, true, true, ~0ull, false);
    std::cout << "CRC Result: " << std::hex << crc << std::endl;
    assert(crc == crc_correct); // Expected CRC value for the example data

    /*


    crc_correct = 0ull;
    crc = 0ull;
    r = 3; 
    k = 8;
    polynomial = 0xB;
    message = {0x01}; // Example data
    std::cout << "===================="
              << "\nStarting CRC Test"
              << "\nr = " << std::dec << r
              << "\nk = " << std::dec << k
              << "\n===================="
              << "\nPolynomial: " << std::hex << polynomial
              << std::endl;
    // print message
    std::cout << "Message: ";
    printVector(message);
    std::cout << "\n====================" << std::endl;

    //set corret value 
    crc_correct = 0x03; 
    crc = CRC::computeCRC(polynomial, r, k, message, false, true, false, true, false);
    std::cout << "CRC Result: " << std::hex << crc << std::endl;
    assert(crc == crc_correct); // Expected CRC value for the example data
    */


    return 0;
}