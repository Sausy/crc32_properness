#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <sstream>
#include <numeric>
#include <bitset>

#include "./include/crc.h"

/**
 * @brief Test class
 * 
 */
class test{
    public:
        test(CRC *c, std::string comment, std::string data_path = "./data/");

        // calculate the hamming distance of a vector
        uint32_t hamming_distance(std::vector<uint8_t> &v1, std::vector<uint8_t> &v2);
        uint32_t hamming_weight(std::vector<uint8_t> &v);

        // calculate the weight distribution of a codeword
        void weight_distribution_direct();

        CRC *c;
        std::string test_file;
};

/**
 * @brief  Calculate the weight distribution of a codeword
 * 
 */
void test::weight_distribution_direct(){
    std::cout << "[INFO] Calculating weight distribution...direct approach" << std::endl;

    const auto const_kBits = this->c->kBits;
    // data vector size of kBits
    std::vector<bool> data(this->c->kBits, 0);

    std::vector<uint32_t> weight_distribution(this->c->nBits, 0);

    for (auto it = 0; it < 2^this->c->kBits; it++){
        // calculate the CRC
        uint32_t crc = this->c->computeCRC(data);

        // combine data and crc
        std::vector<uint8_t> codeword;
        codeword.insert(codeword.end(), data.begin(), data.end());
        // print codeword as hex string 
        std::cout << "Codeword: " << std::hex << std::accumulate(codeword.begin(), codeword.end(), 0) << std::endl;

        // calculate the weight of the codeword
        uint32_t weight = this->hamming_weight(codeword);

        // weight distribution
        weight_distribution[weight] += 1;

        // increment the data vector

    }
}

/**
 * @brief  Calculate the hamming weight of a vector
 * 
 * @param v     vector
 * @return uint32_t weight
 * 
 */
uint32_t test::hamming_weight(std::vector<uint8_t> &v){
    uint32_t weight = std::reduce(v.begin(), v.end());
    return weight;
}

/**
 * @brief Calculate the hamming distance of two vectors
 * 
 * @param v1    vector 1
 * @param v2    vector 2
 * @return uint32_t distance 
 * 
 * @example hamming_distance({1, 0, 1, 1}, {1, 1, 1, 1}) -> 2
 */
uint32_t test::hamming_distance(std::vector<uint8_t> &v1, std::vector<uint8_t> &v2)
{
    uint32_t distance = 0;
    for (int i = 0; i < v1.size(); i++)
    {
        distance += v1[i] ^ v2[i];
    }
    return distance;
}

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

    this->c = c;
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
    
}

int main() {
    std::vector<test> t;
    // test *t = new test(new CRC(16, 32, 0x8005), "direct");


    std::cout << "===================" << std::endl;
    std::cout << "Brootforce codeword weight" << std::endl;

    // CRC 16 test
    //t.push_back(test(new CRC(16, 32, 0x8005), "direct_parityMatrix"));

    // CRC32 test
    //t.push_back(test(new CRC(32, 64, 0xF1922815), "direct"));

    // CRC4 test
    t.push_back(test(new CRC(4, 7, 0xB), "direct"));

    // start running tests
    std::cout << "\nAmount of tests: "<< t.size() << std::endl;
    
    return 0;
}