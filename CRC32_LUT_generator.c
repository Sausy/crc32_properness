/*
*   CRC32_LUT_generator.c
*   This program generates a CRC32 lookup table for given polynoms
*   using reflection.
*/


#include "CRC32_LUT_generator.h"

/*
uint8_t reflect_bits(uint8_t byte) {
    uint8_t reflected = 0;
    for (int i = 0; i < 8; ++i) {
        reflected = (reflected << 1) | (byte & 1);
        byte >>= 1;
    }
    return reflected;
}
*/


/**
 * Reflects the lower bits of a value.
 *
 * @param val The value to reflect.
 * @param num_bits The number of lower bits to reflect.
 * @return The reflected value.
 */
uint32_t reflect(uint32_t val, int num_bits)
{
    uint32_t reflected = 0;
    for (int i = 0; i < num_bits; i++)
    {
        if (val & (1 << i))
            reflected |= (1 << (num_bits - 1 - i));
    }
    return reflected;
}

/**
 * Generates a CRC32 lookup table for a given polynomial
 *
 * Based on http://www.sunshine2k.de/articles/coding/crc/understanding_crc.html
 *
 * @param crcTable The CRC table to generate.
 * @param polynomial The polynomial to use.
 */
void generate_crc32_table(uint32_t crcTable[CRC_TABLE_SIZE], uint32_t polynomial) {
    for (uint32_t dividend = 0; dividend < 256; dividend++) {
        uint32_t curByte = (uint32_t)(dividend << 24);
        for (uint8_t bit = 0; bit < 8; bit++) {
            if ((curByte & 0x80000000) != 0) {
                curByte <<= 1;
                curByte ^= polynomial;
            } else {
                curByte <<= 1;
            }
        }

        crcTable[dividend] = curByte;
    }
}



uint32_t crc32(uint32_t crcTable[CRC_TABLE_SIZE], uint8_t *buf, size_t size) {
    uint32_t crc = ~0U;  // Initialize the CRC value to 0xFFF....
    //for (size_t i = 0; i < len; i++) {
    while (size--) {
        uint8_t byte = buf[size];
        uint32_t idx = ((crc >> 24) ^ reflect(byte, 8)) & 0xFF;
        crc = crcTable[idx] ^ (crc << 8);
    }

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
    uint8_t buf_single[] = {0x01};
    uint8_t buf_single2[] = "hello world";
    uint8_t buf[] = {0x01, 0x02, 0x03, 0x04};

    size_t size = sizeof(buf);
    printf("Size of buf: %u\n", size);

    // Calculate CRC32 value
    uint32_t result_single = crc32(crcTable, buf_single, sizeof(buf_single));
    uint32_t result_single2 = crc32(crcTable, buf_single2, sizeof(buf_single2));

    uint32_t result = crc32(crcTable, buf, size);

    printf("\nsingle 8bit : 0x%08X\nhello world string : 0x%08X\n", result_single, result_single2);

    // Check the expected value
    printf("\n=====================\n");
    printf("CRC32 value: 0x%08X\n", result); // Expected: 0x3D5A6C7D

    
    assert(result == 0xA859F0BE);
}
