#ifndef __ANALYSE_WEIGHT_DISTRIBUTION_H__
#define __ANALYSE_WEIGHT_DISTRIBUTION_H__

#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "CRC32_LUT_generator.h"

#define DATAWORD_LENGTH 32
#define DATAWORD_ARRAY 4  // DATAWORD_LENGTH /8
#define CODEWORD_LENGTH (DATAWORD_LENGTH + CRC_LENGTH)

// get single weight of a codeword
// also known as hamming weight
// max return value of get_hamming_weight is 2^(DATAWORD_LENGTH + CRC_LENGTH)
uint32_t get_hamming_weight(const void *codew, size_t length, uint32_t crc);

// get weight distribution of a codeword
void get_weight_distribution(uint64_t weight_distribution[CODEWORD_LENGTH], uint32_t crcTable[CRC_TABLE_SIZE]);

// Function to increment a big-endian data word
void increment_dataword(uint8_t *dataword, size_t size);

#endif // MACRO
