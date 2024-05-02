#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <sstream>

#include "./include/crc.h"

/**
 * @brief Test class
 * 
 */
class test{
    public:
        test(CRC *c, std::string comment, std::string data_path = "./data/");

        std::string test_file;
};

/**
 * @brief Construct a new test::test object
 * 
 * @param c 
 * @param comment 
 * @param data_path 
 * 
 * @example test::test(new CRC(16, 32, 0x8005), "direct") -> test_file = "CRC16_32bit_0x8005_direct.data"
 */
test::test(CRC *c, std::string comment, std::string data_path){
    std::cout << "[INFO] Data can be found in " << data_path << std::endl;

    /*
    auto nBits = c->nBits;
    auto rBits = c->rBits;
    auto polynomial = c->polynomial;

    this->test_file = "CRC" + std::to_string(rBits) + "_" + std::to_string(nBits) + "bit_";

    char hexString[64 * sizeof(uint8_t) + 1];
    // returns decimal value of hex
    sprintf(hexString, "%x", polynomial);

    this->test_file += "0x" + std::string(hexString) + "_";
    this->test_file += comment;
    this->test_file += ".data";
    */
}

int main() {
    std::cout << "===================" << std::endl;
    std::cout << "Brootforce codeword weight" << std::endl;

    std::vector<test> t;
    //test *t = new test(new CRC(16, 32, 0x8005), "direct");

    // CRC 16 test
    t.push_back(test(new CRC(16, 32, 0x8005), "direct"));

    // CRC32 test
    t.push_back(test(new CRC(32, 64, 0xF1922815), "reflected"));

    // start running tests
    std::cout << "\nAmount of tests: "<< t.size() << std::endl;
    
    return 0;
}