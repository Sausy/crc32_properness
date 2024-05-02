#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <sstream>

#include "./include/crc.h"

/**
 * @brief Get the test name object
 * 
 * @param data_length 
 * @param crc_length 
 * @param polynomial 
 * @param comment 
 * @return std::string 
 * 
 * @example get_test_name(64, 32, 0xF1922815, "direct") -> "CRC32_64bit_0xF1922815_direct.data"
 * 
 * @note This function is used to generate the test name for the data dump file
 */
std::string get_test_name(int data_length, int crc_length, uint32_t polynomial, std::string comment){
    std::string test_name = "CRC" + std::to_string(crc_length) + "_" + std::to_string(data_length) + "bit_";

    char hexString[64 * sizeof(uint8_t) + 1];
    // returns decimal value of hex
    sprintf(hexString, "%x", polynomial);

    test_name += "0x" + std::string(hexString) + "_";
    test_name += comment;
    test_name += ".data";

    return test_name;
}



int main() {
    std::cout << "=========" << std::endl;
    std::cout << "Brootforce codeword weight" << std::endl;

    std::vector<std::string> test;
    // first define Test and data dump name
    uint32_t polynomial = 0xF1922815;
    test.push_back(get_test_name(64, 32, polynomial, "direct"));

    
    std::cout << "Test name: " << test[0] << std::endl;
    
    return 0; 
}