#include "CRC.h"
#include <iostream>

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
    std::cout<<"[INFO] CRC type: " << type << std::endl;
    std::cout<<"[INFO] CRC polynomial: " << polynomial << std::endl;
    std::cout<<"[INFO] CRC initial: " << initial << std::endl;
    std::cout<<"[INFO] CRC reflected: " << reflected << std::endl;
    std::cout<<"[INFO] CRC result reflected: " << resultReflected << std::endl;
    std::cout<<"[INFO] CRC final XOR: " << finalXOR << std::endl;


    this->type = type;
    this->polynomial = polynomial;
    this->initial = initial;
    this->reflected = reflected;
    this->resultReflected = resultReflected;
    this->finalXOR = finalXOR;

    if (type != 8 && type != 16 && type != 32)
        throw std::invalid_argument("Invalid CRC type specified. Allowed values are 8, 16, or 32.");
    
    this->rBits = type;
    this->nBits = n;
    this->kBits = n - type;

    if (this->kBits < 0)
        throw std::invalid_argument("Invalid k must be > 0");

    this->LUT = createLUT(polynomial, type, reflected);
    this->G = generatorMatrix(polynomial, this->rBits, this->kBits);
    this->systematicG = SystematicGeneratorMatrix(polynomial, this->rBits, this->kBits);
    this->H = generateParityCheckMatrix(polynomial, this->rBits, this->kBits); 
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

        switch (type)
        {
        case 8:
            lut[i] = lut[i] & 0xFF;
            break;
        case 16:
            lut[i] = lut[i] & 0xFFFF;
            break;
        default:
            break;
        }
        
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
    if (type != 8 && type != 16 && type != 32)
        throw std::invalid_argument("Invalid CRC type specified.");

    uint32_t shift = type - 8;
    uint32_t output_cleaner = 0xFFFFFFFF >> (32 - type);

    uint32_t crc = initial; // Start with the initial CRC value.
    uint32_t idx = 0; 
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
        crc ^= 0xFFFFFFFF ; // Apply the final XOR mask if specified.
    
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

/*
// Function to multiply matrices
std::vector<uint8_t> multiply(const std::vector<std::vector<uint8_t>> &G, const std::vector<uint8_t> &vec)
{
    std::vector<uint8_t> result(G.size(), 0);
    for (size_t i = 0; i < G.size(); ++i)
    {
        for (size_t j = 0; j < vec.size(); ++j)
        {
            result[i] ^= (G[i][j] & vec[j]);
        }
    }
    return result;
}

// Function to check if vector is zero
bool isZero(const std::vector<uint8_t> &vec)
{
    for (int v : vec)
    {
        if (v != 0)
            return false;
    }
    return true;
}
*/

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
    // int r = 32 - __builtin_clz(polynomial) - 1; // Assuming polynomial is non-zero, compute r as the degree of the polynomial.
    // int n = k + r; // Codeword length.
    //std::vector<std::vector<uint8_t>> G(k, std::vector<uint8_t>(n, 0));
    auto G = generatorMatrix(polynomial, r, k);
 
    int rows = G.size();
    int cols = G[0].size();

    for (int r = 0, lead = 0; r < rows && lead < cols; ++r, ++lead)
    {
        int i = r;
        while (G[i][lead] == 0)
        {
            ++i;
            if (i == rows)
            {
                i = r;
                ++lead;
                if (lead == cols)
                    return G;
            }
        }

        std::swap(G[i], G[r]);

        for (i = 0; i < rows; ++i)
        {
            if (i != r && G[i][lead])
            {
                for (int j = 0; j < cols; ++j)
                {
                    G[i][j] ^= G[r][j];
                }
            }
        }
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
    //int r = 32 - __builtin_clz(polynomial) - 1; // Assuming polynomial is non-zero, compute r as the degree of the polynomial.
    int n = k + r;                              // Codeword length.
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
    //int r = 32 - __builtin_clz(polynomial) - 1; // Compute r as before.
    int n = k + r;

    std::vector<std::vector<uint8_t>> H(r, std::vector<uint8_t>(n, 0));

    auto G = SystematicGeneratorMatrix(polynomial, r, k);

    // Seperate the identity matrix from the generator matrix
    // for simplicity we generate the identity matrix
    std::vector<std::vector<uint8_t>> I_k(k, std::vector<uint8_t>(k, 0));
    // Set the diagonal to be 1s
    for (auto t = 0; t < k; t++)
        I_k[t][t] = 1;

    // Remove the Identiy matrix from G and write it to Matrix P
    std::vector<std::vector<uint8_t>> P(k, std::vector<uint8_t>(n - k, 0));
    for (auto row = 0; row < k; row++)
        for (auto col = 0; col < (n - k); col++)
            P[row][col] = G[row][col + k];

    // Transpose P
    std::vector<std::vector<uint8_t>> P_T(n - k, std::vector<uint8_t>(k, 0));
    for (auto row = 0; row < k; row++)
        for (auto col = 0; col < (n - k); col++)
            P_T[col][row] = P[row][col];

    // Create the parity check matrix H
    // H = [P_T | I_k]
    for (auto row = 0; row < r; row++)
        for (auto col = 0; col < n - k; col++)
            H[row][col] = P_T[row][col];

    for (auto row = 0; row < r; row++)    
        for (auto col = n - k; col < n; col++)
            H[row][col] = I_k[row][col - (n - k)];


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


