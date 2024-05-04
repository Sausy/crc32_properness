#include "common.h"


#include <cassert>


/**
 * Increments a std::vector<uint8_t> by a given amount.
 * The vector is treated as a big-endian representation of an unsigned integer.
 *
 * @param data The vector of uint8_t representing the number to be incremented.
 * @param increment The amount by which to increment the number.
 * @return The incremented vector.
 */
void incVec(std::vector<uint8_t>& data, unsigned int increment) {
    unsigned int carry = increment;
    for (int i = data.size() - 1; i >= 0 && carry > 0; --i) {
        carry += data[i];  // Add the increment and any existing carry to the current byte.
        data[i] = static_cast<uint8_t>(carry % 256);  // Update the byte with the new value, modulo 256.
        carry /= 256;  // Calculate new carry, if any.
    }

    // If carry is still non-zero here, it means the vector was too small to store the entire result
    // This scenario needs proper handling, possibly by resizing the vector and adding new elements at the front.
    // For now, we assume that the vector is sufficiently large to handle the increment.
    // To handle the scenario where the vector is too small, uncomment the following code:
    /*
    while (carry > 0) {
        data.insert(data.begin(), static_cast<uint8_t>(carry % 256));
        carry /= 256;
    }
    */
}

/**
 * @brief  Shifts a vector to the left by a given amount.
 *
 * @param vec The vector to be shifted.
 * @param shift The amount by which to shift the vector.
 * @return The shifted vector.
 * @note The vector is treated as a big-endian representation of an unsigned integer.
 *      The shift operation is equivalent to multiplying the number by 2^shift.
 */
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


/**
 * @brief Print a vector
 * 
 * 
 * @param v  The vector to be printed
 */
void printVector(const std::vector<uint8_t> &v){
    //std::cout << "0x" << std::setw(8) << std::setfill('0') << std::hex; 
    for (auto it = v.begin(); it != v.end(); it++){
        std::cout << std::hex << (int)*it << " ";
    }
    std::cout << std::endl;
}



/**
 * Generates a lookup table for the number of 1s in each possible byte value (0-255).
 *
 * This function creates a lookup table that can be used to quickly determine the number of 1s (bits set to '1')
 * in any byte (8 bits). The table is generated using dynamic programming where the count for each byte is derived
 * from previously calculated values. This makes the function efficient and fast.
 *
 * @return A std::vector<uint8_t> where each index represents a byte value and its value represents
 *         the count of 1s in that byte.
 */
std::vector<uint8_t> generateLookupTable() {
    std::vector<uint8_t> table(256, 0);  // Initialize a vector with 256 zeros.
    for (int i = 0; i < 256; ++i) {
        // The number of 1s in the current byte 'i' can be calculated as:
        // The lowest bit + number of 1s in the rest of the byte.
        // (i & 1) checks if the lowest bit of i is set.
        // table[i / 2] gives the count of 1s in i shifted right by one bit (i.e., i/2).
        table[i] = (i & 1) + table[i / 2];
    }
    return table;
}

/** XORs two vectors element-wise and returns the result.
 *
 * @param v1 The first vector to be XORed.
 * @param v2 The second vector to be XORed.
 * @return The resulting vector after XOR operation.
 */
std::vector<uint8_t> xorVectors(std::vector<uint8_t> &v1, std::vector<uint8_t> &v2){
    assert(v1.size() == v2.size());

    std::vector<uint8_t> result(v1.size(), 0);

    for (uint32_t i = 0; i < v1.size(); i++){
        
        result[i] = v1[i] ^ v2[i];
        assert(v1[i] <= 1);
        assert(v2[i] <= 1);
        assert(result[i] <= 1);
    }
    return result;
}
