#ifndef __CRC32_LUT_GENERATOR_H__
#define __CRC32_LUT_GENERATOR_H__

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdio.h>

#define CRC_LENGTH 32
#define CRC_TABLE_SIZE 256


// Reflects Input
uint8_t reflect_bits(uint8_t byte);

// Reflects the lower bits of a value.
uint32_t reflect(uint32_t val, int num_bits);

// Generates a CRC32 lookup table for a given polynomial using reflection.
void generate_crc32_table(uint32_t crcTable[CRC_TABLE_SIZE], uint32_t polynomial);

// print the CRC table
void print_CRC32_table(uint32_t crcTable[CRC_TABLE_SIZE]); 

// return crc32 value based on the crcTable
uint32_t crc32(uint32_t crcTable[CRC_TABLE_SIZE], const void *buf, size_t size);

//=======================================================
// Unit tests
//=======================================================
void test_all_crc();

// Test function for generate_crc32_table
void test_generate_crc32_table();

// Test function for crc32
void test_crc32_expected_value();

#endif // MACRO
