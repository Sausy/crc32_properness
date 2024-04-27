/*
*   CRC32_LUT_generator.c
*   This program generates a CRC32 lookup table for given polynoms
*   using reflection.
*/


#include "CRC32_LUT_generator.h"

//
uint8_t reflect_bits(uint8_t byte) {
    uint8_t reflected = 0;
    for (int i = 0; i < 8; ++i) {
        reflected = (reflected << 1) | (byte & 1);
        byte >>= 1;
    }
    return reflected;
}

/**
 * Reflects the lower bits of a value.
 *
 * @param val The value to reflect.
 * @param num_bits The number of lower bits to reflect.
 * @return The reflected value.
 */
uint32_t reflect(uint32_t val, int num_bits)
{
    uint32_t result = 0;
    for (int i = 0; i < num_bits; i++)
    {
        if (val & (1 << i))
            result |= (1 << (num_bits - 1 - i));
    }
    return result;
}

/**
 * Generates a CRC32 lookup table for a given polynomial using reflection.
 *
 * @param crcTable The table to be filled with CRC32 values.
 * @param polynomial The polynomial used for generating the CRC32 values.
 */
void generate_crc32_table(uint32_t crcTable[CRC_TABLE_SIZE], uint32_t polynomial)
{
    polynomial = reflect(polynomial, 32); // Reflect the polynomial
    for (uint32_t i = 0; i < CRC_TABLE_SIZE; i++)
    {
        uint32_t crc = reflect(i, 8); // Reflect the input byte
        for (uint8_t j = 0; j < 8; j++){
            if (crc & 1){
                crc = (crc >> 1) ^ polynomial;
            }else{
                crc >>= 1;
            }
        }
        crcTable[i] = reflect(crc, 32); // Reflect the result before storing it
    }
}

// Current implementation 
// * Reflects Input 
// * Reflects Output
// * Does not invert resulting CRC
uint32_t crc32(uint32_t crcTable[CRC_TABLE_SIZE], const void *buf, size_t size) {
    const uint8_t *p = buf;
    uint32_t crc = ~0U;  // Initialize the CRC value to 0xFFF....
    uint8_t idx = 0; 

    /*
    crc = ~0U; // Initialize the CRC value to 0xFFF....
    while (size--)
        crc = crcTable[(crc ^ *p++) & 0xFF] ^ (crc >> 8);

    return crc ^ ~0U;
    */
   
    while (size--){
        //idx = ( (crc >> 24) ^ *p) & 0xFF; //Only 8Bit-MSB are used for the table lookup id
        //since we want the input to be reflected
        idx = ((crc >> 24) ^ reflect_bits(p[size])) & 0xFF;
        crc = crcTable[idx] ^ (crc << 8);
        p++;
    }

    // we also defined in robbus that the crc should be reflected on the output 
    crc = reflect(crc, 32); 
    
    // invert the crc value 
    // easiest way is to xor 0xffff....
    return crc ^ ~0U;
}

// Print the CRC table
void print_CRC32_table(uint32_t crcTable[CRC_TABLE_SIZE]){
    printf("Table Size = %u\n", CRC_TABLE_SIZE);
    printf("crcTable = ");
    for (int i = 0; i < CRC_TABLE_SIZE; i++){
        printf("[0x%08X]", crcTable[i]);

        if (i % 8 == 1) {
            printf("\n");
        }
    }
    printf("\n\n");
}


//==============================================================================
// Unit Tests

void test_all_crc() {
    //test_generate_crc32_table();
    test_crc32_expected_value();
}

void test_generate_crc32_table() {
    uint32_t crcTable[CRC_TABLE_SIZE];
    uint32_t polynomial = 0xF4ACFB13;

    generate_crc32_table(crcTable, polynomial);

    // Check the first few values in the CRC table
    assert(crcTable[0] == 0x00000000);
    assert(crcTable[1] == 0xF4ACFB13);
    assert(crcTable[2] == 0x1DF50D35);
    assert(crcTable[3] == 0xE959F626);
    assert(crcTable[4] == 0x3BEA1A6A);
    assert(crcTable[5] == 0xCF46E179);
    assert(crcTable[6] == 0x261F175F);
}



void test_crc32_expected_value() {
    // Initialize CRC table
    uint32_t crcTable[CRC_TABLE_SIZE];
    uint32_t polynomial = 0xF4ACFB13;
    generate_crc32_table(crcTable, polynomial);
    print_CRC32_table(crcTable);

    // Initialize input buffer
    const uint8_t buf_single[] = {0x01};
    const uint8_t buf_single2[] = {0x80};
    const uint8_t buf[] = {0x01, 0x02, 0x03, 0x04};
    const uint8_t buf2[] = {0x80, 0x40, 0xC0, 0x02};
    const uint8_t buf3[] = {0x04, 0x03, 0x02, 0x01};
    const uint8_t buf4[] = {0x02, 0xC0, 0x40, 0x80};
    // const uint8_t buf[] = {0x01};
    // const uint8_t buf[] = {0x01};
    // const char buf[] = "Hello, world!";
    size_t size = sizeof(buf);
    printf("Size of buf: %u\n", size);

    // Calculate CRC32 value
    uint32_t result_single = crc32(crcTable, buf_single, sizeof(buf_single));
    uint32_t result_single2 = crc32(crcTable, buf_single2, sizeof(buf_single2));

    uint32_t result = crc32(crcTable, buf, size);
    uint32_t result2 = crc32(crcTable, buf2, sizeof(buf2));
    uint32_t result3 = crc32(crcTable, buf3, sizeof(buf3));
    uint32_t result4 = crc32(crcTable, buf4, sizeof(buf4));

    printf("\n single 8bit : 0x%08X\nsingle 8bit : 0x%08X\n", result_single, result_single2);

    // Check the expected value
    printf("\n=====================\n");
    printf("CRC32 value: 0x%08X\n", result); // Expected: 0x3D5A6C7D
    printf("CRC32 value: 0x%08X\n", result ^ 0xFFFFFFFF);  // Expected: 0x3D5A6C7D
    printf("\n=======byte wise reflected =======\n");
    printf("CRC32 value: 0x%08X\n", result2);  // Expected: 0x3D5A6C7D
    printf("CRC32 value: 0x%08X\n", result2 ^ 0xFFFFFFFF);  // Expected: 0x3D5A6C7D
    printf("\n=======0x04, 0x03, 0x02, 0x01=======\n");
    printf("CRC32 value: 0x%08X\n", result3);               // Expected: 0x3D5A6C7D
    printf("CRC32 value: 0x%08X\n", result3 ^ 0xFFFFFFFF);  // Expected: 0x3D5A6C7D
    printf("\n=========fully reflexed input =======\n");
    printf("CRC32 value: 0x%08X\n", result4);               // Expected: 0x3D5A6C7D
    printf("CRC32 value: 0x%08X\n", result4 ^ 0xFFFFFFFF);  // Expected: 0x3D5A6C7D
    assert(result == 0xAF6C2C93);
}
