#include "CRC.h"
#include <iostream>
#include <cassert>
#include "common.h"
#include "crc.h"
#include <bitset>



int main()
{

    uint64_t crc_correct = 0ull;
    uint64_t crc = 0ull;
    auto r = 3;
    auto k = 4;
    uint64_t polynomial = 0b011;

    bool conf_crcPoly_reflect = false;
    bool conf_inReflect = false;
    bool conf_outReflect = true;
    bool conf_outXor = false;
    uint64_t conf_init = 0ull;
    

    std::vector<uint8_t> message = {0b0001}; // Example data
    std::cout << "\n\n===================="
              << "\nStarting CRC Test"
              << "\nr = " << std::dec << r
              << "\nk = " << std::dec << k
              << "\ninit: " << std::hex << conf_init
              << "\nReflex IN: " << std::dec << conf_inReflect
              << "\nReflex OUT: " << std::dec << conf_outReflect
              << "\nXOR OUT: " << std::dec << conf_outXor
              << "\nPolynomial: " << std::hex << polynomial
              << std::endl;
    // print message
    std::cout << "Message: ";
    printVector(message);
    std::cout << "====================" << std::endl;
    crc = CRC::computeCRC(polynomial, r, k, message, conf_crcPoly_reflect, conf_inReflect, conf_outReflect, conf_outXor, conf_init, false);
    std::cout << "CRC Result: " << std::hex << crc << std::endl;
    assert(crc == 0b110); // Expected CRC value for the example data
    

    message = {0b0010}; // Example data
    std::cout << "\n===================="
              << "\nStarting CRC Test"
              << "\nr = " << std::dec << r
              << "\nk = " << std::dec << k
              << "\ninit: " << std::hex << conf_init
              << "\nReflex IN: " << std::dec << conf_inReflect
              << "\nReflex OUT: " << std::dec << conf_outReflect
              << "\nXOR OUT: " << std::dec << conf_outXor
              << "\nPolynomial: " << std::hex << polynomial
              << std::endl;
    // print message
    std::cout << "Message: ";
    printVector(message);
    std::cout << "====================" << std::endl;
    crc = CRC::computeCRC(polynomial, r, k, message, conf_crcPoly_reflect, conf_inReflect, conf_outReflect, conf_outXor, conf_init, false);
    std::cout << "CRC Result: " << std::hex << crc << std::endl;
    assert(crc == 0b011); // Expected CRC value for the example data

    message = {0b0100}; // Example data
    std::cout << "\n===================="
              << "\nStarting CRC Test"
              << "\nr = " << std::dec << r
              << "\nk = " << std::dec << k
              << "\ninit: " << std::hex << conf_init
              << "\nReflex IN: " << std::dec << conf_inReflect
              << "\nReflex OUT: " << std::dec << conf_outReflect
              << "\nXOR OUT: " << std::dec << conf_outXor
              << "\nPolynomial: " << std::hex << polynomial
              << std::endl;
    // print message
    std::cout << "Message: ";
    printVector(message);
    std::cout << "====================" << std::endl;
    crc = CRC::computeCRC(polynomial, r, k, message, conf_crcPoly_reflect, conf_inReflect, conf_outReflect, conf_outXor, conf_init, false);
    std::cout << "CRC Result: " << std::hex << crc << std::endl;
    assert(crc == 0b111); // Expected CRC value for the example data

    message = {0b1000}; // Example data
    std::cout << "\n===================="
              << "\nStarting CRC Test"
              << "\nr = " << std::dec << r
              << "\nk = " << std::dec << k
              << "\ninit: " << std::hex << conf_init
              << "\nReflex IN: " << std::dec << conf_inReflect
              << "\nReflex OUT: " << std::dec << conf_outReflect
              << "\nXOR OUT: " << std::dec << conf_outXor
              << "\nPolynomial: " << std::hex << polynomial
              << std::endl;
    // print message
    std::cout << "Message: ";
    printVector(message);
    std::cout << "====================" << std::endl;
    crc = CRC::computeCRC(polynomial, r, k, message, conf_crcPoly_reflect, conf_inReflect, conf_outReflect, conf_outXor, conf_init, false);
    std::cout << "CRC Result: " << std::hex << crc << std::endl;
    assert(crc == 0b101); // Expected CRC value for the example data

    //=======================
    std::vector<uint8_t> unit_vector = {0x00, 0x00, 0x00, 0x01}; // Example data
    std::vector<uint8_t> datvector = CRC::unitVectorToDatVector(unit_vector);
    //print datvector
    std::cout << "datvector: ";
    printVector(datvector);
    
    std::cout << "\n===================="
              << "\nStarting CRC Test"
              << "\nr = " << std::dec << r
              << "\nk = " << std::dec << k
              << "\ninit: " << std::hex << conf_init
              << "\nReflex IN: " << std::dec << conf_inReflect
              << "\nReflex OUT: " << std::dec << conf_outReflect
              << "\nXOR OUT: " << std::dec << conf_outXor
              << "\nPolynomial: " << std::hex << polynomial
              << std::endl;
    // print message
    std::cout << "====================" << std::endl;
    std::cout << "Message: ";
    printVector(unit_vector);
    crc = CRC::computeCRC(polynomial, r, k, datvector, conf_crcPoly_reflect, conf_inReflect, conf_outReflect, conf_outXor, conf_init, false);
    std::cout << "CRC Result: " << std::hex << crc << std::endl;

    // I think the shift is wrong here
    // after i 4*8 = 32, the shift should result in a 1
    std::cout << "\n====\nk = " << std::dec << k << std::endl;
    //auto loop_up = ;
    for(auto i = 0; i < (unit_vector.size()*8+r); i+=8){
        //k = 4 * 8; 
        uint64_t shift_data = CRC::shift_data_from_vec(unit_vector, 8, i);
        std::cout << "Example 1 Result: 0x" << std::hex << shift_data << std::endl;
        std::cout   << "[" << std::dec << (int)i 
                    << "] \t Shifted data: " 
                    << std::hex << shift_data 
                    << "\t" << std::bitset<64>(shift_data)                  
                    << std::endl;
    }
    
    assert(crc == 0b110); // Expected CRC value for the example data

    /*
    // ================
    unit_vector = {0x00, 0x00, 0x01, 0x00};
    std::cout << "====================" << std::endl;
    std::cout << "Message: ";
    printVector(unit_vector);
    crc = CRC::computeCRC(polynomial, r, k, unit_vector, conf_crcPoly_reflect, conf_inReflect, conf_outReflect, conf_outXor, conf_init, true);
    std::cout << "CRC Result: " << std::hex << crc << std::endl;
    */

    /*
    // ================
    std::cout << "====================" << std::endl;
    unit_vector = {0x00, 0x01, 0x00, 0x00};
    std::cout << "Message: ";
    printVector(unit_vector);
    crc = CRC::computeCRC(polynomial, r, k, unit_vector, conf_crcPoly_reflect, conf_inReflect, conf_outReflect, conf_outXor, conf_init, true);
    std::cout << "CRC Result: " << std::hex << crc << std::endl;
    */

    // ================
    unit_vector = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01};
    datvector = CRC::unitVectorToDatVector(unit_vector);
    k = 16; 
    r = 6; 
    
    polynomial = 0b000111;
    std::cout << "====================" << std::endl;
    std::cout << "Message: ";
    printVector(unit_vector);
    crc = CRC::computeCRC(polynomial, r, k, unit_vector, conf_crcPoly_reflect, conf_inReflect, conf_outReflect, conf_outXor, conf_init, false);
    std::cout << "CRC Result: " << std::hex << crc << std::endl;


    
    return 0;
}