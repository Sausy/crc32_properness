#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <sstream>
#include <numeric>
#include <bitset>
#include <cmath>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <thread>
#include <future>
#include <iomanip>
#include <stdexcept>
#include <cassert>

#include <map>

#include "./include/crc.h"
#include "./include/common.h"

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
        void weight_distribution_directParityMatrix();

        CRC *c;
        std::string test_file;

        void run_tests();
    
    private:
        void store_data(const std::vector<uint32_t> &distribution);
        std::vector<uint8_t> hamming_weight_LUT;
        std::filesystem::path fo = "./data/";

        const std::vector<std::string> select_brootforce_method{"direct", "direct_parityMatrix"};
        std::string selected_method = "direct";
        // function pointer to different testing methods
        // dictionary that shows function pointer based on the method string e.g.: direct
        std::map<std::string, void (test::*)()> method_dict{
            {"direct", &test::weight_distribution_direct},
            {"direct_parityMatrix", &test::weight_distribution_directParityMatrix}
        };

};

/**
 * @brief Construct a new test::test object
 * 
 * @param c         CRC object
 * @param comment   Comment to be added to the file name
 * @param data_path Path to the data folder
 */
test::test(CRC *c, std::string comment, std::string data_path)
{
    this->fo = data_path;
    std::cout << "[INFO] Data can be found in " << data_path << std::endl;

    this->c = c;
    auto nBits = c->nBits;
    auto rBits = c->rBits;
    auto polynomial = c->polynomial;

    this->test_file = "CRC" + std::to_string(rBits) + "_" + std::to_string(nBits) + "bit_";

    char hexString[64 * sizeof(uint8_t) + 1];
    // returns decimal value of hex
    sprintf(hexString, "%x", polynomial);

    // set polynomial
    this->test_file += "0x" + std::string(hexString) + "_";

    // set initial value
    sprintf(hexString, "%x", c->initial);
    this->test_file += "0x" + std::string(hexString) + "_";

    // set input reflected
    this->test_file += c->reflected ? "1_" : "0_";

    // set result reflected
    this->test_file += c->resultReflected ? "1_" : "0_";

    // set final XOR
    this->test_file += c->finalXOR ? "1_" : "0_";

    // if comment is part of select_brootforce_method
    if (method_dict.find(comment) != method_dict.end()){
        this->test_file += comment;
        this->selected_method = comment;
    }else{
        this->selected_method = "direct"; // default method
        throw std::invalid_argument("comment is not part of select_brootforce_method");
        this->test_file += select_brootforce_method[0];
    }
    this->test_file += ".data";

    // generate the lookup table
    this->hamming_weight_LUT = generateLookupTable();
}

/**
 * @brief Runs the tests for the `test` class.
 */
void test::run_tests(){
    // run selected function pointer 
    (this->*method_dict[this->selected_method])();
}



/**
 * @brief Calculate the weight distribution of a codeword using the direct parity matrix method.
 * 
 * This function analyzes the weight distribution of a given data set using the direct parity matrix method.
 * The direct parity matrix method is a technique used to determine the distribution of weights in a data set.
 * It calculates the number of occurrences of each weight value and stores the results in an array.
 * This function does not return any value.
 * 
 * @param c A pointer to the CRC object to be analyzed.
 * @param comment A string containing additional information about the test.
 * @param data_path A string containing the path to the data folder.
 * 
 */
void test::weight_distribution_directParityMatrix(){
    std::cout << "\n[INFO] Calculating weight distribution...direct VIA Parity Matrix" << std::endl;

    // the code is currently limited to data that is a multiple of 8
    /*
    if (this->c->kBits % 8 != 0){
        // throw error
        throw std::invalid_argument("kBits is not a multiple of 8");
        return;
    }
    */

    std::cout << "Parity check Matrix for CRC" << this->c->type << std::endl;
    printMatrix(this->c->H);

    //---------------------------------
    uint32_t weight = 0;
    std::vector<uint32_t> weight_distribution(this->c->nBits + 1, 0);

    // get systematic parity check matrix
    auto H = this->c->H;
    // get number of rows and columns of matrix H
    uint32_t rows = H.size();
    uint32_t cols = H[0].size();

    // size must be [n-k, n]
    assert(rows == (uint32_t)(this->c->nBits - this->c->kBits));
    assert(cols == (uint32_t)this->c->nBits);

    std::cout << "Cols = " << cols << " Rows = " << rows << std::endl;

    // number of subsets
    uint64_t subsets = (uint64_t)pow((long double)2, (long double)rows);

    // initialize a row vector  
    std::vector<uint8_t> v(cols, 0);

    // to track the time
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto chunk_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    uint64_t chunk_size = 1000000;

    std::cout << "[INFO] Total subsets: " << std::dec << subsets << std::endl;

    // Iterate over all possible subsets (excluding the empty set, starting from 1)
    for(uint64_t it = 1; it <= subsets; it++){
        // reset the vector to zero
        v = std::vector<uint8_t>(cols, 0);

        // The bit pattern represents the assamble of 
        // the subset vectors that must be combined via XOR
        // to get the codeword
        // e.g.: 0b0011 
        // means that the 1st and 2nd vector must be XORed
        // to get the codeword
        std::bitset<64> bit_pattern(it);
    

        //std::cout << "\n=======\n";
        // iterate over the rows of the parity check matrix
        // to bitwise XOR the vectors
        for (uint32_t i = 0; i < rows; i++){
            // if the bit is set to 1
            if (bit_pattern[i] == 1){
                // XOR the vectors
                //std::cout<< "vec: \t"; 
                //printVector(v); 
                //std::cout<< "H[" << i << "]: \t";
                //printVector(H[i]);
                v = xorVectors(v, H[i]);

                //std::cout << "res: \t";
                //printVector(v);
            }
        }
        
        //printout Vector
        printVector(v);

        // calculate the weight of the codeword
        weight = this->hamming_weight(v);
        //std::cout << "weight: " << std::dec << weight << std::endl;

        // weight distribution
        weight_distribution[weight] += 1;

        // For debugging purposes
        // tracing a chunk shall reflect the time left 
        if(it % chunk_size == 0 && it != 0){
            t2 = std::chrono::high_resolution_clock::now();
            chunk_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
            double chunks_left = (subsets - it) / chunk_size;
            double time_left = (chunks_left * chunk_duration.count()) / 1000;
            double time_left_min = time_left / 60;
            double time_left_hours = time_left_min / 60;
            std::cout << "[INFO] chunk execution took : "
                      << std::dec << chunk_duration.count()
                      << " milliseconds\tChunks left: "
                      << chunks_left
                      << std::endl;

            std::cout << "[INFO] " << std::dec << it << " out of " << subsets
                      << " done. "
                      << std::dec
                      << time_left
                      << " seconds left ["
                      << time_left_hours
                      << " hours]"
                      << std::endl;

            t1 = std::chrono::high_resolution_clock::now();
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t0);
    std::cout << "[INFO] Done! " << std::dec << duration.count() << " milliseconds" << std::endl;

    // print the weight distribution
    for (uint32_t i = 0; i < weight_distribution.size(); i++){
        std::cout << "Weight " << std::dec << i << ": " << std::dec << weight_distribution[i] << std::endl;
    }

    // store the data in a file
    this->store_data(weight_distribution);
}

/**
 * @brief  Calculate the weight distribution of a codeword
 *
 * The weight distribution is calculated by generating all possible data
 * and calculating the weight of the codeword. The weight of the codeword
 * is the number of bits set to 1 in the codeword.
 *
 * The weight distribution is stored in a file
 *
 */
void test::weight_distribution_direct()
{
    std::cout << "\n[INFO] Calculating weight distribution...direct approach" << std::endl;

    // the code is currently limited to data that is a multiple of 8
    if (this->c->kBits % 8 != 0){
        //throw error
        throw std::invalid_argument("kBits is not a multiple of 8");
        return;
    }

    // data vector size of k-Bits/8
    std::vector<uint8_t> data((uint32_t)this->c->kBits/8, 0);
    // the weight distribution must be n+1 because weight 0 reflects 
    // the empty set. which is not needed but part of it
    std::vector<uint32_t> weight_distribution(this->c->nBits + 1, 0);
    std::vector<uint8_t> crc_vector((uint32_t)this->c->rBits/8, 0);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto chunk_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    uint32_t weight = 0;
    uint32_t crc = 0;

    uint64_t total = (uint64_t)pow((long double)2, (long double)this->c->kBits);

    uint64_t chunk_size = 1000000;

    for (uint64_t it = 0; it < total; it++)
    {   
        // calculate the CRC
        crc = this->c->computeCRC(data);

        switch(this->c->rBits){ 
            case 8:
                crc_vector[0] = (uint8_t)crc;
                break;
            case 16:
                crc_vector[0] = (uint8_t)(crc >> 8);
                crc_vector[1] = (uint8_t)(crc & 0xFF);
                break;
            case 32:
                crc_vector[0] = (uint8_t)(crc >> 24);
                crc_vector[1] = (uint8_t)(crc >> 16);
                crc_vector[2] = (uint8_t)(crc >> 8);
                crc_vector[3] = (uint8_t)(crc & 0xFF);
                break;
            default:
                throw std::invalid_argument("rBits is not 8, 16 or 32");
                return;
        }

        // concatenate the data and the crc
        // to get the codeword = {data, crc}
        std::vector<uint8_t> codeword = concatenateVectors(data, crc_vector);

        // calculate the weight of the codeword
        weight = this->hamming_weight(codeword);

        // weight distribution
        weight_distribution[weight] += 1;

        // For debugging purposes
        // tracing a chunk shall reflect the time left 
        if(it % chunk_size == 0 && it != 0){
            t2 = std::chrono::high_resolution_clock::now();
            chunk_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
            double chunks_left = (total - it) / chunk_size;
            double time_left = (chunks_left * chunk_duration.count()) / 1000;
            double time_left_min = time_left / 60;
            double time_left_hours = time_left_min / 60;
            std::cout << "[INFO] chunk execution took : "
                      << std::dec << chunk_duration.count()
                      << " milliseconds\tChunks left: "
                      << chunks_left
                      << std::endl;

            std::cout << "[INFO] " << std::dec << it << " out of " << total
                      << " done. "
                      << std::dec
                      << time_left
                      << " seconds left ["
                      << time_left_hours
                      << " hours]"
                      << std::endl;

            t1 = std::chrono::high_resolution_clock::now();
        }

        // increment the data vector
        incVec(data, 1);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t0);
    std::cout << "[INFO] Done! " << std::dec << duration.count() << " milliseconds" << std::endl;
    

    // store the data in a file
    this->store_data(weight_distribution);
}

/**
 * @brief  Store the data in a file
 * 
 * @param distribution  The weight distribution to be stored.
 * 
 * The data shall be stored in a Json format to additionall store 
 * important information about the test.
 * e.g.: CRC type, polynomial, method, etc.
 * 
 */
void test::store_data(const std::vector<uint32_t> &distribution){
    std::cout << "[INFO] Storing data in " << this->test_file << std::endl;
    std::filesystem::path filePath = this->fo / this->test_file;
    // std::ofstream file;
    //  Open the file using ofstream in trunc mode to delete content and create if it doesn't exist
    std::ofstream file(filePath.string(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

    if (!file) {
        // If the file could not be opened, output an error message.
        std::cerr << "Error opening file: " << this->test_file << std::endl;
        return;
    }
    std::cout << "[INFO] File opened successfully" << std::endl;

    // Start Json 
    file << "{\n";

    // print default infomation
    file << "\t" << "\"CRC\": " << std::dec << this->c->type << ",\n";
    file << "\t" << "\"Polynomial\": " << std::hex << "\"0x" << this->c->polynomial << "\",\n";
    file << "\t" << "\"Initial\": " << std::hex << "\"0x" << this->c->initial << "\",\n";
    file << "\t" << "\"InputReflected\": " << std::boolalpha << this->c->reflected << ",\n";
    file << "\t" << "\"ResultReflected\": " << std::boolalpha << this->c->resultReflected << ",\n";
    file << "\t" << "\"FinalXOR\": " << std::boolalpha << this->c->finalXOR << ",\n";
    file << "\t" << "\"Method\": \"" << this->selected_method << "\",\n";
    file << "\t" << "\"kBits\": " << std::dec << this->c->kBits << ",\n";
    file << "\t" << "\"rBits\": " << std::dec << this->c->rBits << ",\n";
    file << "\t" << "\"nBits\": " << std::dec << this->c->nBits << ",\n";
    file << "\t" << "\"Data\": [\n";

    // write the data to the file
    for (auto &d: distribution){
        file << "\t\t"<< std::dec << d; 
        if (&d != &distribution.back())
            file << ",";
        
        file << "\n";
    }
    // End Data
    file << "\t" << "],\n";

    /*

    // add Systematic Generator Matrix
    file << "\t" << "\"Systematic_Generator_Matrix\": [\n";
    for (auto &row: this->c->systematicG){
        file << "\t\t[";
        for (auto &col: row){
            file << std::dec << (int)col;
            if (&col != &row.back())
                file << ",";
        }
        file << "]";
        if (&row != &this->c->systematicG.back())
            file << ",";
        file << "\n";
    }
    file << "\t" << "],\n";

    // add parity check matrix
    file << "\t" << "\"Parity_Check_Matrix\": [\n";
    for (auto &row: this->c->H){
        file << "\t\t[";
        for (auto &col: row){
            file << std::dec << (int)col;
            if (&col != &row.back())
                file << ",";
        }
        file << "]";
        if (&row != &this->c->H.back())
            file << ",";
        file << "\n";
    }
    file << "\t" << "],\n";
    */


    // Add a timestamp to the file
    // first in Unix time
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    file << "\t" << "\"Unix_Timestamp\": " << std::dec << now_c << ",\n";
    // second in human readable format
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %X");
    file << "\t" << "\"Timestamp\": \"" << ss.str() << "\",\n";

    // add editor information
    file << "\t" << "\"Editor\": \"Sausy\"\n";



    // End Json
    file << "}\n";

    // Close the file after writing the data.
    file.close();

    std::cout << "[INFO] Done!" << std::endl;
}

/**
 * Calculates the Hamming weight (number of bits set to 1) of a given vector of uint8_t.
 *
 * This function uses a precomputed lookup table to find the Hamming weight of each byte in the vector efficiently.
 * By using a lookup table, the function avoids the need to count bits manually for each byte, significantly improving performance.
 *
 * @param data A constant reference to a std::vector<uint8_t> containing the data to compute the Hamming weight for.
 * @return The total Hamming weight (uint32_t) of all bytes in the vector.
 */
uint32_t test::hamming_weight(std::vector<uint8_t> &v){
    // Access the static lookup table generated previously.
    // It is declared static so that it is created only once and reused, which saves computation time.

    uint32_t count = 0; // Initialize the count of 1s found.
    for (uint8_t byte : v)
    {
        // Add the count of 1s in 'byte', as found in the lookup table.
        count += this->hamming_weight_LUT[byte];
    }
    return count; // Return the total count of 1s in all bytes.
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
    for (uint32_t i = 0; i < (uint32_t)v1.size(); i++)
        distance += v1[i] ^ v2[i];

    return distance;
}



int main() {
    std::vector<test> t;
    // TODO:
    // 1. Make time prediction better -> sine a single run of direct_parityMatrix is not as long as the next one 
    // 2. Add more CRC types
    // 3. Add more methods


    std::cout << "===================" << std::endl;
    std::cout << "Brootforce codeword weight" << std::endl;

    /*
    t.push_back(test(new CRC(3, 7, 0xB, 0U, false, false, false), "direct_parityMatrix"));
    t.push_back(test(new CRC(3, 7, 0xB, 0U, false, true, false), "direct_parityMatrix"));
    t.push_back(test(new CRC(3, 7, 0xB, 0U, true, false, false), "direct_parityMatrix"));
    t.push_back(test(new CRC(3, 7, 0xB, 0U, true, true, false), "direct_parityMatrix"));
    */
    //t.push_back(test(new CRC(8, 16, 0x31, 0U, true, true, false), "direct_parityMatrix"));

    /*
    // CRC8 test
    t.push_back(test(new CRC(8, 16, 0x31), "direct"));
    t.push_back(test(new CRC(8, 16, 0x31, 0U, true, true, false), "direct"));
    t.push_back(test(new CRC(8, 16, 0x2F, ~0U, false, false, true), "direct"));

    // CRC8 parity check matrix test
    t.push_back(test(new CRC(8, 16, 0x31), "direct_parityMatrix"));
    t.push_back(test(new CRC(8, 16, 0x31, 0U, true, true, false), "direct_parityMatrix"));
    t.push_back(test(new CRC(8, 16, 0x2F, ~0U, false, false, true), "direct_parityMatrix"));

    // CRC16 test
    t.push_back(test(new CRC(16, 32, 0x8005), "direct"));
    t.push_back(test(new CRC(16, 32, 0x8005, 0U, true, true, true), "direct")); //CRC16-MAXIM
    t.push_back(test(new CRC(16, 32, 0x1021, 0x89EC, true, true, false), "direct")); // CRC16-TMS37157

    // CRC16 parity check matrix test
    t.push_back(test(new CRC(16, 32, 0x8005), "direct_parityMatrix"));
    t.push_back(test(new CRC(16, 32, 0x8005, 0U, true, true, true), "direct_parityMatrix"));      // CRC16-MAXIM
    t.push_back(test(new CRC(16, 32, 0x1021, 0x89EC, true, true, false), "direct_parityMatrix")); // CRC16-TMS37157

    // =======================
    // Long Runners

    // CRC32 test
    t.push_back(test(new CRC(32, 64, 0xF1922815), "direct"));
    t.push_back(test(new CRC(32, 64, 0xF4ACFB13), "direct"));
    t.push_back(test(new CRC(32, 64, 0x04C11DB7), "direct"));

    // direct_parityMatrix
    t.push_back(test(new CRC(32, 64, 0xF1922815), "direct_parityMatrix"));
    t.push_back(test(new CRC(32, 64, 0xF4ACFB13), "direct_parityMatrix"));
    t.push_back(test(new CRC(32, 64, 0x04C11DB7), "direct_parityMatrix"));

    // 128 bit data length
    t.push_back(test(new CRC(32, 128, 0xF1922815), "direct_parityMatrix"));
    */

    // custom length
    //t.push_back(test(new CRC(32, 48, 0xF1922815), "direct_parityMatrix"));

    /*

    t.push_back(test(new CRC(32, 64, 0xF1922815, 0x00000000, false, false, false), "direct"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0x00000000, false, false, true), "direct"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0x00000000, false, true, false), "direct"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0x00000000, false, true, true), "direct"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0x00000000, true, false, false), "direct"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0x00000000, true, false, true), "direct"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0x00000000, true, true, false), "direct"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0x00000000, true, true, true), "direct"));

    t.push_back(test(new CRC(32, 64, 0xF1922815, 0xFFFFFFFF, false, false, false), "direct"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0xFFFFFFFF, false, false, true), "direct"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0xFFFFFFFF, false, true, false), "direct"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0xFFFFFFFF, false, true, true), "direct"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0xFFFFFFFF, true, false, false), "direct"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0xFFFFFFFF, true, false, true), "direct"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0xFFFFFFFF, true, true, false), "direct"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0xFFFFFFFF, true, true, true), "direct"));
    */

    t.push_back(test(new CRC(32, 64, 0xF1922815, 0U, false, false, false), "direct_parityMatrix"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0U, false, false, true), "direct_parityMatrix"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0U, false, true, false), "direct_parityMatrix"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0U, false, true, true), "direct_parityMatrix"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0U, true, false, false), "direct_parityMatrix"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0U, true, false, true), "direct_parityMatrix"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0U, true, true, false), "direct_parityMatrix"));
    t.push_back(test(new CRC(32, 64, 0xF1922815, 0U, true, true, true), "direct_parityMatrix"));

    // start running tests
    std::cout << "\nAmount of tests: "<< t.size() << std::endl;
    for (auto &testPtr: t){
        testPtr.run_tests();
    }
    
    return 0;
}