#include "CRC.h"
#include <iostream>
#include "common.h"

/**
 * Constructor to initialize the CRC object.
 * 
 * @param type The type of CRC (e.g., 8, 16, 32 bits).
 * @param polynomial The polynomial used for CRC calculations, specified as a hex value.
 * @param initial The initial value for CRC calculations.
 * @param reflected Whether the input data is reflected.
 * @param resultReflected Whether the result is reflected.
 * @param finalXOR Whether to XOR the final result with 0xFFFFFFFF.
 * 
 * 
 */
CRC::CRC(int type, int n, uint32_t polynomial, uint32_t initial, bool reflected, bool resultReflected, bool finalXOR){
    
    std::cout<<"\n[INFO] CRC object created. " << std::endl;
    std::cout<<"[INFO] CRC type: r=" << std::dec << type << std::endl;
    std::cout<<"[INFO] CRC polynomial: " << std::hex << "0x" << polynomial << std::endl;
    std::cout<<"[INFO] CRC initial: " << std::hex << initial << std::endl;
    std::cout<<"[INFO] CRC reflected: "<< std::dec << reflected << std::endl;
    std::cout<<"[INFO] CRC result reflected: " << resultReflected << std::endl;
    std::cout<<"[INFO] CRC final XOR: " << finalXOR << std::endl;


    this->type = type;
    this->polynomial = polynomial;
    this->initial = initial;
    this->reflected = reflected;
    this->resultReflected = resultReflected;
    this->finalXOR = finalXOR;

    //if (type != 8 && type != 16 && type != 32)
    //    throw std::invalid_argument("Invalid CRC type specified. Allowed values are 8, 16, or 32.");
    
    this->rBits = type;
    this->nBits = n;
    this->kBits = n - type;

    if (this->kBits < 0)
        throw std::invalid_argument("Invalid k must be > 0");
    
    // generate a LUT that is not reflected
    this->LUT = createLUT(polynomial, type, false);

    this->G = generatorMatrix(polynomial, this->rBits, this->kBits);
    this->systematicG = this->SystematicGeneratorMatrix(polynomial, this->rBits, this->kBits);
    this->H = this->generateParityCheckMatrix(polynomial, this->rBits, this->kBits);
}

/**
 * Creates a lookup table (LUT) for calculating CRC using a specified polynomial.
 * 
 * @param polynomial The polynomial used in the CRC calculation, specified as a hex value.
 * @param type The CRC type (e.g., 8, 16, 32 bits).
 * @param reflected Whether the LUT should use reflected (reverse of normal) bit ordering.
 * @return A vector containing the computed CRC lookup table.
 * 
 * Mathematical Background:
 * - A CRC is a type of checksum calculation used to detect errors in data storage or transmission.
 * - The 'polynomial' used here defines the CRC polynomial employed for the calculation, represented in binary.
 * - The CRC process treats the segment of data as a large polynomial, divides it by the 'polynomial',
 *   and takes the remainder as the CRC. The polynomial division here is simulated using binary XOR operations.
 * 
 * Implementation Details:
 * - The LUT expedites the CRC calculation by precomputing results for all possible 8-bit sequences.
 * - For each byte (0 to 255), compute its CRC by initially setting it as the lower 8 bits of an 'n'-bit register
 *   (where 'n' is the CRC type, e.g., 16 for CRC-16), then performing a bitwise calculation.
 * - Each bit of the byte is processed: if the highest bit of the register is set, the register is XORed
 *   with the polynomial (simulating polynomial subtraction which is equivalent to XOR in binary).
 * - This loop simulates the shifting operation of polynomial division, where each bit is checked and the polynomial
 *   is subtracted conditionally based on the most significant bit.
 * - The result is stored in the LUT, potentially reflected if required by the CRC specification.
 * 
 * This function enables quick CRC calculations during actual data processing by using this precomputed LUT.
 */
std::vector<uint32_t> CRC::createLUT(uint32_t polynomial, int type, bool reflected) {
    std::vector<uint32_t> lut(256, 0);
    uint64_t mask = 0ull; // 0xFFFFFFFF >> (32 - type);
    mask = mask | ((~0ull >> uint64_t(64 - type)));

    for (int i = 0; i < 256; ++i) {
        uint32_t crc = (reflected) ? reflect(i, 8) : i;
        crc <<= type - 8; // Prepare the byte for processing in an 'n'-bit register.
        for (int j = 0; j < 8; j++) {
            // Perform the modulo-2 division by checking the highest bit and XORing with the polynomial if set.
            if (crc & (1 << (type - 1)))
                crc = (crc << 1) ^ polynomial;
            else
                crc <<= 1;
        }
        lut[i] = reflected ? reflect(crc, type) : crc;

        // to ensure that the length of the crc is correct
        lut[i] = lut[i] & mask;
    }
    return lut;
}


/**
 * Reflects the lower 'nBits' bits of 'data'.
 * 
 * @param data The data whose bits are to be reflected.
 * @param nBits The number of lower bits in 'data' to reflect.
 * @return The reflection of the lower 'nBits' bits of 'data'.
 * 
 * Mathematical Background:
 * - Bit reflection is used in some CRC algorithms to match the bit order of data transmissions,
 *   which may not align with typical byte-oriented processing in computing environments.
 * - Reflection inverts the order of bits, meaning the bit received or processed first becomes the last in the sequence.
 *   This is important when the transmission protocol sends the least significant bit first.
 * 
 * Implementation Details:
 * - This function takes the specified number of bits from 'data', inverts their order, and returns the new value.
 * - The operation is performed using a loop which checks each bit and places it in the reversed position in the result.
 */
uint32_t CRC::reflect(uint32_t data, int nBits) {
    uint32_t reflection = 0;
    for (int i = 0; i < nBits; ++i) {
        if (data & 0x01) {
            reflection |= (1 << ((nBits - 1) - i));
        }
        data >>= 1;
    }
    return reflection;
}

/**
 * Computes the CRC value for a given set of data using the specified lookup table (LUT).
 *
 * @param lut The lookup table generated for a specific polynomial and CRC type.
 * @param data The data for which the CRC is to be computed, given as a vector of bytes.
 * @param initial The initial value of the CRC register before the computation starts.
 * @param inputReflected Specifies if the input bytes should be processed in reflected (bit-reversed) order.
 * @param resultReflected Specifies if the output CRC should be reflected (bit-reversed).
 * @param finalXOR Specifies if the final CRC value should be XORed with a final value (commonly 0xFFFFFFFF).
 * @return The computed CRC value as a 32-bit unsigned integer.
 *
 * Mathematical Background:
 * - The function computes the CRC by processing each byte of input data through a series of table lookups and bitwise operations.
 * - The CRC computation model can be seen as applying a mask (derived from the polynomial) onto the data as it is fed into a shift register.
 * - Depending on the inputReflected and resultReflected flags, the function adjusts the bit order of the input and output to match specific protocol requirements.
 *
 * Implementation Details:
 * - The CRC computation involves initializing a register (crc), then updating this register for each byte of the input data based on the LUT.
 * - For each byte, the index into the LUT is determined by XORing the byte (possibly reflected) with the current CRC value's upper byte.
 * - The crc variable is then updated using the value from the LUT. This simulates the division of data by the polynomial, capturing the remainder.
 * - After processing all data bytes, if the CRC result needs to be reflected, this is done before applying a final XOR mask if specified.
 */
uint32_t CRC::computeCRC(const std::vector<uint32_t> &lut, const std::vector<uint8_t> &data,
                         int type, 
                         uint32_t initial, bool inputReflected,
                         bool resultReflected, bool finalXOR)
{
    uint64_t output_cleaner = 0ull; // 0xFFFFFFFF >> (32 - type);
    output_cleaner = output_cleaner | ((~0ull >> uint64_t(64 - type)));

    // Todo make this more flexible
    // by allowing crc types between r [1, 64]  
    if (type != 8 && type != 16 && type != 32)
        throw std::invalid_argument("Ror now Invalid CRC type specified. needs to be more flexible");

    uint32_t shift = type - 8;
    uint32_t crc = initial; // Start with the initial CRC value.
    uint32_t idx = 0; 

    //std::cout << "type: " << std::dec << type << std::endl;
    //std::cout << "shift: " << std::dec << shift << std::endl;
    //std::cout << "lut[1]: " << std::hex << lut[1] << std::endl;
    //std::cout << "data[0]" << std::hex << data[0] << std::endl;


    for (auto byte : data)
    {
        if (inputReflected)
            byte = reflect(byte, 8); // Reflect the byte if required.
        // Update the CRC by looking up the transformation in the LUT.
        // The index is the XOR of the current crc's upper byte and the data byte.
        idx = ((crc >> shift) ^ byte) & 0xFF;
        crc = lut[idx] ^ (crc << 8);
        //crc = (crc >> 8) ^ lut[(crc ^ byte) & 0xFF];
    }
    if (resultReflected)
        crc = reflect(crc, type); // Reflect the final CRC value if needed.
    if (finalXOR)
        crc ^= ~0ull; // Apply the final XOR mask if specified.

    crc = crc & output_cleaner;

    return crc;
}

/**
 * @brief Computes the CRC value for a given set of data using the specified lookup table (LUT).
 * 
 * @param data The data for which the CRC is to be computed, given as a vector of bytes.
 * @return The computed CRC value as a 32-bit unsigned integer.
 * 
 * This function is an overload of the static computeCRC function, using the object's parameters.
 * 
 */
uint32_t CRC::computeCRC(const std::vector<uint8_t> &data){
    return computeCRC(this->LUT, data, this->type, this->initial, this->reflected, this->resultReflected, this->finalXOR);
}

// Function to multiply a polynomial by x^shift (equivalent to left shifting)
std::vector<int> multiplyByXPower(const std::vector<int>& poly, int shift) {
    std::vector<int> result(poly.size() + shift, 0);
    for (uint32_t i = 0; i < poly.size(); ++i) {
        result[i + shift] = poly[i];
    }
    return result;
}

// Function to add two polynomials in GF(2)
std::vector<int> addPolynomials(const std::vector<int>& poly1, const std::vector<int>& poly2) {
    uint32_t max_size = (uint32_t)std::max(poly1.size(), poly2.size());
    std::vector<int> result(max_size, 0);

    for (uint32_t i = 0; i < max_size; ++i) {
        int a = (i < poly1.size()) ? poly1[i] : 0;
        int b = (i < poly2.size()) ? poly2[i] : 0;
        result[i] = a ^ b; // XOR operation
    }

    return result;
}

// Function to reduce a polynomial by another polynomial in GF(2)
std::vector<int> reducePolynomial(const std::vector<int>& poly, const std::vector<int>& generator) {
    std::vector<int> result = poly;
    int deg_gen = generator.back();

    while (result.size() > 0 && result.back() >= deg_gen) {
        int shift = result.back() - deg_gen;
        std::vector<int> temp = multiplyByXPower(generator, shift);
        result = addPolynomials(result, temp);
        while (result.size() > 0 && result.back() == 0) {
            result.pop_back();
        }
    }

    return result;
}

uint64_t CRC::shift_data_from_vec(const std::vector<uint8_t>& data, int k, int shift) {
    // Start with a uint64_t to accumulate the bits
    uint64_t result = 0;

    // Load bits from vector to the result considering the vector as big-endian
    for (size_t i = 0; i < data.size(); ++i) {
        // Shift left by 8 each time as we add a new byte, then OR with the new byte
        result = (result << 8) | data[i];
    }

    // Adjust result based on bit length k
    // Remove the bits above k by shifting left to remove excess bits and then shifting back right
    if (k < 64) {
        result = (result << (64 - k)) >> (64 - k);
    }

    // Apply the shift: align the desired bits to the least significant bits of the result
    if (shift > 0) {
        // Calculate how many bits to shift right to get the desired output
        int shift_amount = k - shift;
        result = (result >> shift_amount) & ((1ULL << shift) - 1);
    } else {
        result = 0; // If shift is zero, return zero
    }

    return result;
}


std::vector<uint8_t> CRC::unitVectorToDatVector(const std::vector<uint8_t>& unit_vector) {
    uint64_t temp = 0;
    std::vector<uint8_t> data_vector;
    uint16_t offset = 0; 
    
    for (int i = unit_vector.size() - 1; i >= 0; --i) {
        //temp <<= 1;
        temp |= unit_vector[i] << offset;

        offset++;

        if(i%8 == 0){
            data_vector.insert(data_vector.begin(), temp);
            offset = 0; 
            //data_vector.push_back(static_cast<uint8_t>(temp));
            temp = 0ull;
        }

        
    }
    
    
    return data_vector;
}



/**
 * @brief Generates the generator matrix for a given CRC polynomial and dataword length.
 * 
 * @param polynomial The polynomial used in the CRC calculation, specified as a hex value.
 * @param r The length of the CRC.
 * @param message The length of the dataword. 
 * @param conf_crcPoly_reflect  
 * @param conf_inReflect 
 * @param conf_outReflect 
 * @param conf_outXor  
 * @return uint64_t The CRC value.
 * 
 * Mathematical Background:
 * - The generator matrix G in coding theory is used to generate the codewords from the datawords.
 * - The matrix is structured such that multiplying it by a dataword vector (in binary) will result in a codeword vector.
 * - The rows of G represent the coefficients of the polynomial that multiplies the data bits to form the codeword.
 * - For CRC, the generator matrix is constructed by appending the identity matrix of size k and the matrix representation of the polynomial.
 * 
 */
uint64_t CRC::computeCRC(   uint64_t polynomial, 
                            uint8_t r,  
                            uint16_t k, 
                            const std::vector<uint8_t>& message,
                            bool conf_crcPoly_reflect,
                            bool conf_inReflect, bool conf_outReflect, bool conf_outXor,
                            uint64_t conf_init, 
                            bool conf_inputIsBitVecotr) {
    if (r < 1 || r > 64) {
        throw std::invalid_argument("r must be within the range [1,64]");
    }

    std::vector<uint8_t> data; 
    if (conf_inputIsBitVecotr) {
        data = unitVectorToDatVector(message);
    }else{
        // reflect the input message if required
        data = message;
    }
    conf_inputIsBitVecotr = false;

    // to ensure that the crc_register only hols 
    // data that is r bits long
    uint64_t mask = 0ull; // 0xFFFFFFFF >> (32 - type);
    mask = mask | ((~0ull >> uint64_t(64 - r)));

    // crc shift register
    // here is trap init value must be
    // put infront of the data 
    // and is not the initial value of the crc_register
    uint64_t crc_reg = 0ull; //~0uLL; 

    bool flag_MSB_one = false;

    
    // if message is a bit vector 
    // the bit data must be shifted to 
    // a vector
    /*
    uint8_t vector_pointer = (uint8_t)(r - 1);
    uint8_t pos_counter = 0 ;
    if(conf_inputIsBitVecotr){
        // reset the data vector
        data = std::vector<uint8_t>(r, 0);
        pos_counter = 0; 

        for(auto i = message.size(); i > 0; i--){
            if(pos_counter % 8 == 0 && pos_counter != 0)
                vector_pointer--;
            // i%8 will have results in the range of [0,7]
            data[vector_pointer] |= (message[i-1] & 0x01) << (int)(pos_counter % 8);
            pos_counter++; 
        }
    }
    */

    // if the input is a bit vector the bit length is the size of the vector
    uint32_t loop_max = k + r;
    
    //if(conf_inputIsBitVecotr)
    //    loop_max = (message.size() + r + 1) * 8;

    if (conf_inReflect){
        // TODO: make this more flexible 
        std::cout << "[INFO] Reflecting input data" << std::endl;
        for (auto& byte : data){
            //hacky workaround will fail for e.g.: k = 15
            if(k < 8)
                byte = reflect(byte, k);  // Reflect each byte of the input message if required.
            else
                byte = reflect(byte, 8);  // Reflect each byte of the input message if required.
        }
    }

    // print out data 
    std::cout << "[INFO] Data: ";
    printVector(data);


    // Process each byte in the message.
    // e.g.: for a 3bit message it takes 4 iterations
    // crc-reg       message = 0b111
    // [0, 0 , 0] | (0b111 >> 3) = 0b000
    // [0, 0 , 0] | (0b111 >> 2) = 0b001
    // [0, 0 , 0] | (0b111 >> 1) = 0b011
    // [0, 0 , 0] | (0b111 >> 0) = 0b111
    // which is eqaul to i in 0 <= i <= 3 
    // where 3 equals the size of the message

    std::cout << "k: " << (int)(k) << std::endl;
    std::cout << "loops: " << (int)(loop_max) << std::endl;

    //uint32_t loop_up = 1; 
    //if (conf_inputIsBitVecotr)
    //    loop_up = 8;

    for (uint32_t i = 1; i <= (uint32_t)(loop_max); i++) {
        // reset flag 
        flag_MSB_one = false;

        // enxure the crc_reg is has only bits 
        // in the length of r 
        crc_reg = crc_reg & mask;

        // check if MSB is "1" and therfor if it will be shifted 
        // out in the next shift operation
        //std::cout << "[INFO] shift: " << (int)(r-1) << "\tcrc_reg: " << std::hex << (int)(crc_reg >> (r-1)) << std::endl;
        if (((crc_reg >> (r-1)) & 1ULL) == 1)
            flag_MSB_one = true;

        // shift the crc_reg by 1
        crc_reg <<= 1;  
        // load next bit from the message
        //if (conf_inputIsBitVecotr)
        //    crc_reg |= (shift_data_from_vec(data, 8, i-1) & 1ull);
        //else 
        //    crc_reg |= (shift_data_from_vec(data, k, i) & 1ull);

        crc_reg |= (shift_data_from_vec(data, k, i) & 1ull);

        /*
        std::cout << "[" << std::dec << (int)(i) << "] crc_reg: " << std::bitset<6>(crc_reg) << "\t";
        std::cout << "shift: " << std::hex << (int)(shift_data_from_vec(data, k, i))
                  << "\t" << std::bitset<64>(shift_data_from_vec(data, k, i))
                  << std::endl;

        if (conf_inputIsBitVecotr){
            //std::cout << "[" << std::dec << (int)i << "] crc_reg: " << std::hex << crc_reg << "\t";
            std::cout << "[" << std::dec << (int)(i-1) << "] crc_reg: " << std::bitset<4>(crc_reg) << "\t";
            std::cout << "shift: " << std::hex << (int)(shift_data_from_vec(data, 8, i - 1))
                      << "\t" << std::bitset<64>(shift_data_from_vec(data, 8, i - 1))
                      << std::endl;
        }
        */

        // if MSB was "1" then XOR the polynomial
        if (flag_MSB_one){
            crc_reg = (crc_reg ^ polynomial) & mask;
            //std::cout << "[INFO] XOR operation \t reg: " << std::hex << crc_reg << std::endl; 
        }
    }

    // Final XOR operation after all bits processed.
    if (conf_outReflect)
        crc_reg = reflect(crc_reg, r); // Reflect the CRC result if required.

    if (conf_outXor)
        crc_reg ^= ((1ULL << r) - 1); // Perform final XOR operation.

    crc_reg = crc_reg & mask; // mask the crc value to the r bits

    return crc_reg; // Return the final CRC value.
}

/**
 * @brief Generates the generator matrix for a given CRC polynomial and dataword length.
 * 
 * @param message 
 * @param polynomial 
 * @param r 
 * @param k 
 * @return uint64_t 
 */
uint64_t CRC::calculateCRC_unitVec(const std::vector<uint8_t>& message, uint64_t polynomial, int r, int k) {
    // Combine the message into a single large integer for processing.
    uint64_t data = 0;
    for (auto byte : message) {
        data = (data << 8) | byte;
    }

    // Append r zero bits to the right of the data.
    data <<= r;

    // Polynomial division: Extract the top bit position of the polynomial.
    //uint64_t highest_bit = 1ull << (r - 1);

    // Perform polynomial division (bitwise division).
    for (int i = k + r - 1; i >= r; i--) {
        if (data & (1ull << i)) {  // Check if the ith bit is set.
            data ^= (polynomial << (i - r));  // Subtract the polynomial shifted left.
        }
    }

    //reverse the data
    // make the following maybe configurable   
    // data = reflect(data, r);   // not needed for current implementation 
    
    
    //
    data ^= ~0ull; // XOR with F  

    // mask 
    uint64_t mask = 0ull; 
    // depending on the r value the mask is set
    mask = mask | ((~0ull >> uint64_t(64-r))); 
    std::cout << "[" << std::dec << r << "]mask: " << std::hex << mask << std::endl;

    // The remaining lower r bits are the CRC.
    return data & mask;
}

// Function to convert uint32_t polynomial to std::vector<int> representation
std::vector<int> convertToGenerator(uint32_t polynomial, int r) {
    std::vector<int> generator;
    for (int i = 0; i < r; ++i) {  // Check all 32 bits in a uint32_t
        if (polynomial & (1 << i)) {  // Check if the i-th bit is set
            generator.push_back(i);
        }
    }
    generator.push_back(r); // Set the r-th bit to 1
    return generator;
}

// TODO: Thsi function is actually in common.h defined i dont know why it 
// cannot find it 
// workaround is to copy the function here
// TODO: this function suddenly stops shifting after 9bits
std::vector<uint8_t> BitShiftVector_v2(const std::vector<uint8_t> &vec, uint8_t shift) {
    // shift value to the left 
    // if bit is shifted over the 8bit boundary it is added to the next byte
    std::vector<uint8_t> shifted_vec(vec.size(), 0);
    std::vector<uint8_t> buffer = vec;

    if (shift == 0) {
        return vec;
    }

    for (auto s = 0; s < shift; ++s){
        for (int i = (int)vec.size(); i > 0; i--) {
            shifted_vec[i] = (buffer[i] << 1) & 0xFF;

            if(((buffer[i+1] | 0x7F) == 0xFF) && (i < (int)(vec.size() - 1))){
                shifted_vec[i] |= 0x01;
            }
        }
        buffer = shifted_vec;
    }
    
    return shifted_vec;
}

// @brief shift the unit vector by a given amount
// @param vec the unit vector
// @param shift the amount to shift the unit vector
// @return the shifted unit vector
std::vector<uint8_t> CRC::shift_unitVector(uint8_t size, uint8_t shift) {
    // shift value to the left 
    // if bit is shifted over the 8bit boundary it is added to the next byte
    std::vector<uint8_t> shifted_vec(size, 0);

    shifted_vec[size - shift - 1] = 1;

    return shifted_vec;
}

/**
 * Generates the generator matrix for a given CRC polynomial and dataword length.
 *
 * @param polynomial The polynomial used in the CRC calculation, specified as a hex value.
 * @param k The length of the dataword.
 * @return A matrix (vector of vectors) representing the generator matrix of size [k, n] where n = k + r.
 *
 * Mathematical Background:
 * - The generator matrix G in coding theory is used to generate the codewords from the datawords.
 * - The matrix is structured such that multiplying it by a dataword vector (in binary) will result in a codeword vector.
 * - The rows of G represent the coefficients of the polynomial that multiplies the data bits to form the codeword.
 * - For CRC, the generator matrix is constructed by appending the identity matrix of size k and the matrix representation of the polynomial.
 *
 * Example
 * The results are following 4×7 generator matrix:

        d1	d2	d3	d4  p1	p2	p3

        1	0	0	0   0	1	1
G = 	0	1	0	0   1	0	1	  = [ I_k | -A^T  ]
        0	0	1	0   1	1	0
        0	0	0	1   1	1	1
 *
 * Implementation Details:
 * - This function constructs the matrix by appending the identity matrix I_k to a matrix representation of the CRC polynomial shifted to align with the CRC bits.
 */
std::vector<std::vector<uint8_t>> CRC::SystematicGeneratorMatrix(uint32_t polynomial, int r, int k)
{
    // n codeword (aka.: telegram) length
    // k dataword length
    // r crc length
    std::cout << "[INFO] Generating systematic generator matrix" << std::endl;
    std::cout << "[INFO] k: " << std::dec << k << std::endl;
    std::cout << "[INFO] r: " << std::dec << r << std::endl;

    int n = k + r; // Codeword length.
    // G = [ I_k | -A^T ]
    // means G has the size [k, n]
    // Initialize the generator matrix with zeroes
    std::vector<std::vector<uint8_t>> G(k, std::vector<uint8_t>(n, 0));
    
    // set the identity matrix
    for (int i = 0; i < k; ++i)
        G[i][i] = 1;

    //
    std::vector<uint8_t> unit_vector(k, 0);
    //std::vector<uint8_t> v1(k, 0);
    std::vector<uint8_t> p(r,0);
    //v1[k-1] = 1; 

    uint64_t crc = 0; 

    std::cout   << "\n====================\nStarting with shifting through the unit vectors" 
                << std::endl; 

    // loop through all possible unit vectors
    for (uint8_t i = 0; i < (uint8_t)k; ++i){
        // reset the parity vector
        p = std::vector<uint8_t>(r, 0);

        // TODO the bit shift algorithmus has a bug
        // it only shifts for the first 8bit correctly
        //unit_vector = BitShiftVector_v2(v1, i); // this should be BitShiftVector but it cannot be found
        unit_vector = shift_unitVector(k, i);

        // print the unit vector
        std::cout << "unit vector: ";
        for (auto m : unit_vector) {
            std::cout << std::hex << (int)m << " ";
        }
        std::cout << std::endl;

        // calculate the crc for the unit vector
        //crc = calculateCRC_unitVec(unit_vector, (uint64_t)polynomial, r, k);
        crc = computeCRC(polynomial, r, (uint16_t)k, unit_vector, false, this->reflected, this->resultReflected, this->finalXOR, 0ull, true);

        std::cout << "[INFO] [poly:0x" << std::hex << polynomial  << "]\tCRC value: " << std::hex << crc << "\tk: " << std::dec << k <<std::endl;
        

        //set bit mask as parity vector
        for (auto j = 0; j < r; ++j){
            p.push_back((crc >> j) & 1);
            G[i][j + k] = (crc >> j) & 1;
        }
        

    }

    // print the generator matrix
    std::cout << "Generator Matrix: " << std::endl;
    for (auto row : G) {
        for (auto col : row) {
            std::cout << std::hex << (int)col << " ";
        }
        std::cout << std::endl;
    }
    

    /*
    Example for (7,4)
    P.s.: since r=3 this means x^2 is the highest degree of the polynomial
    g(x) = 1+x+x3

    xi      g(x)qi(x)           di(x)   xi+di(x)
    --------------------------------------
    x3      (1+x+x3)·1          1+x     1+x+x3
    x4      (1+x+x3)·x          x+x2    x+x2+x4
    x5      (1+x+x3)·(1+x2)     1+x+x2  1+x+x2+x5
    x6      (1+x+x3)·(1+x+x3)   1+x2    1+x2+x6
    */

    return G;


    // init the identity matrix
    //for (int i = 0; i < k; ++i)
    //    G[i][i] = 1;

    /*
    // Generate the parity vector based on the CRC polynomial
    std::vector<std::vector<uint8_t>> P(k, std::vector<uint8_t>(r, 0));
    for (int i = 0; i < k; ++i) {
        std::vector<uint8_t> unitVector(k, 0);
        unitVector[i] = 1; // Set the i-th position to 1 to represent the unit vector for the i-th message bit
        P[i] = generateParityVector(unitVector, polynomial, r); // Generate the i-th column of the parity matrix
    }
    */

    /*
    G = generatorMatrix(polynomial, r, k);

    // Perform Gaussian elimination with partial pivoting
    for (int i = 0; i < k; ++i) {
        // Pivot: matrix[i][i] should be 1
        if (G[i][i] == 0) {
            // Find a row below the current row to swap with
            bool found = false;
            for (int row = i + 1; row < k; ++row) {
                if (G[row][i] == 1) {
                    std::swap(G[i], G[row]);
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "Failed to find a necessary pivot. The G may not be transformable into systematic form." << std::endl;
                return {};
            }
        }

        // Zero out all other entries in this column
        for (int row = 0; row < k; ++row) {
            if (row != i && G[row][i] == 1) {
                // XOR current row with the pivot row to make G[row][i] zero
                for (int j = 0; j < n; ++j) {
                    G[row][j] ^= G[i][j];
                }
            }
        }
    }
    */
    

}


/**
 * Generates the generator matrix for a given CRC polynomial and dataword length.
 *
 * @param polynomial The polynomial used in the CRC calculation, specified as a hex value.
 * @param k The length of the dataword.
 * @return A matrix (vector of vectors) representing the generator matrix of size [k, n] where n = k + r.
 *
 * Mathematical Background:
 * - The generator matrix G in coding theory is used to generate the codewords from the datawords.
 * - The matrix is structured such that multiplying it by a dataword vector (in binary) will result in a codeword vector.
 * - The rows of G represent the coefficients of the polynomial that multiplies the data bits to form the codeword.
 * - For CRC, the generator matrix is constructed by appending the identity matrix of size k and the matrix representation of the polynomial.
 *
 * Implementation Details: 
 * - This function constructs the matrix by appending the identity matrix I_k to a matrix representation of the CRC polynomial shifted to align with the CRC bits.
 */
std::vector<std::vector<uint8_t>> CRC::generatorMatrix(uint32_t polynomial, int r, int k)
{
    std::cout << "[INFO] Generating Generator Matrix" << std::endl;

    // The size of the generator matrix is [k, n] where n = k + r.
    int n = k + r;  // Codeword length.
    std::vector<std::vector<uint8_t>> G(k, std::vector<uint8_t>(n, 0));

    uint32_t polynomial_buffer; 

    for (int row_cnt = 0; row_cnt < k; row_cnt++){
        polynomial_buffer = polynomial;
        for(int poly_cnt = 0; poly_cnt <= r; poly_cnt++){
            if(polynomial_buffer & 1)
                G[row_cnt][poly_cnt + row_cnt] = 1;

            polynomial_buffer >>= 1;
        }
    }
    
    return G;
}

/**
 * Generates the parity check matrix for a given CRC polynomial and dataword length.
 *
 * @param polynomial The polynomial used in the CRC calculation, specified as a hex value.
 * @param k The length of the dataword.
 * @return A matrix (vector of vectors) representing the parity check matrix of size [r, n] where n = k + r.
 *
 * Mathematical Background:
 * - The parity check matrix H is used to detect errors in codewords. In coding theory, it's structured such that H * c^T = 0 for a valid codeword c.
 * - For CRC-based systems, the matrix reflects the shift and structure imposed by the CRC polynomial on the codeword bits.
 *
 * Example:
 *  for polynom 0xB for CRC4, the parity check matrix is:
 *  $$h(x) = (1+x^2+x3)(1+x) = 1+x+x^2+x^4$$
 *
 * Implementation Details:
 * - This matrix is complementary to the generator matrix and ensures that valid codewords result in a zero vector when multiplied by this matrix.
 * - The structure typically involves placing the polynomial coefficients in the rightmost part of the matrix and filling the rest with the identity matrix.
 */
std::vector<std::vector<uint8_t>> CRC::generateParityCheckMatrix(uint32_t polynomial, int r, int k)
{
    std::cout <<"[INFO] Generating Parity Check Matrix" << std::endl;
    //int r = 32 - __builtin_clz(polynomial) - 1; // Compute r as before.
    int n = k + r;

    std::vector<std::vector<uint8_t>> H(r, std::vector<uint8_t>(n, 0));

    // Generate the systematic generator matrix G
    auto G = this->SystematicGeneratorMatrix(polynomial, r, k);

    // Load Identiy Matrix to H
    // amount of rows is r=n-k
    // H = [ P | I ]
    // H[n-k, n]
    // Since P [n-k, k] we need to add an offset of k to the column
    for (auto t = 0; t < r; t++)
        H[t][t + k] = 1;    

    // G = [ I | P_t ] 
    // G[k, n]
    // this means I = [k, k]
    // P_t has the size [k, n-k]
    std::vector<std::vector<uint8_t>> P(r, std::vector<uint8_t>(k, 0));
    for (auto row = 0; row < k; row++)
        for (auto col = 0; col < r; col++)
            P[col][row] = G[row][col + k]; // col and row are swaped to directly get the transposed

    // Create the parity check matrix H
    // H = [ P | I]
    for (auto row = 0; row < r; row++)
        for (auto col = 0; col < k; col++)
            H[row][col] = P[row][col];


    return H;
}

/**
 * Prints a CRC lookup table (LUT).
 *
 * @param lut The lookup table generated for CRC calculations.
 * @param columns The number of columns in the printout for better readability.
 *
 * This function displays each entry in the lookup table along with its index. It formats the output
 * in a table-like structure to make the LUT visually comprehensible. The 'columns' parameter allows
 * the output to be split into multiple columns, improving readability for large tables.
 */
void printLUT(const std::vector<uint32_t> &lut, int columns)
{
    //int columnWidth = 10; // Adjust as needed for alignment
    std::cout << "Lookup Table:" << std::endl;
    for (size_t i = 0; i < lut.size(); ++i)
    {
        std::cout << "0x" << std::setw(8) << std::setfill('0') << std::hex << lut[i];
        if ((i + 1) % columns == 0)
            std::cout << std::endl;
        else
            std::cout << " ";
    }
    if (lut.size() % columns != 0) // Ensure there's a newline if the last row isn't full
        std::cout << std::endl;
}


