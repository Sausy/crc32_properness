#include "Analyse_weight_distribution.h"

// get single weight of a codeword
// also known as hamming weight
uint32_t get_hamming_weight(const void *codew, size_t length,  uint32_t crc) {
    uint32_t weight = 0;
    //uint16_t lenght_frame = length + 32;
    const uint8_t *p = codew;

    uint8_t bit_buffer; 
    for (size_t j = 0; j < length; j++) {
        bit_buffer = *p++;

        for (int i = 0; i < 8; i++) {
            weight += (bit_buffer & 0x01);
            bit_buffer = bit_buffer >> 1;
        }
    }

    for (int i = 0; i < 32; i++) {
        weight += (crc & 0x01);
        crc = crc >> 1;
    }
    
    return weight;
}

// Function to increment a big-endian data word
void increment_dataword(uint8_t *dataword, size_t size) {
    for (int i = size - 1; i >= 0; i--) {
        if (++dataword[i] != 0) {  // Increment and check for overflow
            break;                 // No carry, break the loop
        }
        // If there's an overflow (byte was 0xFF and became 0x00), continue to carry to next byte
    }
}

uint8_t is_dataword_full(uint8_t *dataword, size_t size) {   
    for (int i = 0; i < size; i++) {
        if (dataword[i] != 0xFF) {
            return 0;
        }
    }
    return 1;
}

void get_weight_distribution(uint64_t weight_distribution[CODEWORD_LENGTH], uint32_t crcTable[CRC_TABLE_SIZE]){
    
    clock_t begin = clock();
    clock_t t1 = clock();
    clock_t t2 = clock();
    double time_left; 

    printf("\n[INFO]: %ld Calculating weight distribution...\n", begin);

    //uint64_t codeword = 0;
    //uint32_t dataword = 0;
    uint8_t dataword[DATAWORD_ARRAY]; 
    uint32_t crc = 0;
    uint64_t w = 0;

    //uint64_t permutaion = pow(2,DATAWORD_LENGTH);

    //printf("[INFO]: loop max: %"PRIu64"\n", permutaion);

    // 10000000
    //for (uint64_t i = 0; i < permutaion; i++) {
    uint64_t i = 0;
    while (is_dataword_full(dataword, sizeof(dataword)) != 1) {
        crc = crc32(crcTable, dataword, sizeof(dataword));
        //printf("CRC32 value: 0x%08X\n", crc);
        //codeword = ((uint64_t)dataword << CRC_LENGTH) | crc;

        // max return value of get_hamming_weight is 2^(DATAWORD_LENGTH + CRC_LENGTH)
        w = get_hamming_weight(dataword, sizeof(dataword), crc);  // get the weight of a single codeword
        // printf("Weight: %"PRIu64"\n", w);
        weight_distribution[w]++;

        increment_dataword(dataword, sizeof(dataword));

        
        if(i == 10000000){
            i = 0; 
            t2 = clock();
            time_left = (double)(t2 - t1) / CLOCKS_PER_SEC;
            //printf("[INFO]: %"PRIu64" \t/ %"PRIu64" \ttime passed: %lf\n", i, permutaion, time_left);
            printf("time passed: %lf\n", time_left);
            t1 = clock();
        }
        
        i++;
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("[INFO]: %ld Time to calculate weight distribution: %f \n", end, time_spent);
}