#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define POLYNOM 0x1F1922815ULL
#define N 64
#define K 32
#define ROWS (N - K)
#define NUM_SUBSETS (1ULL << ROWS)

// Function to initialize the CRC32 parity check matrix
void init_crc32_parity_check_matrix(uint64_t H[ROWS][N]) {
    for (size_t j = 0; j < N; j++) {
        H[0][j] = (POLYNOM >> (N - 1 - j)) & 1;  // Initialize the first row
    }

    for (size_t i = 1; i < ROWS; i++) {
        // Circular shift of the previous row to the right
        H[i][0] = H[i - 1][N - 1];
        for (int j = 1; j < N; j++) {
            H[i][j] = H[i - 1][j - 1];
        }
    }
    /*
    uint32_t crc_poly = 0xF1922815;
    
    for (int i = 0; i < POLY_LEN; i++) {
        uint64_t shifted_poly = (uint64_t)crc_poly << i; 
        for (int j = 0; j < TOTAL_LEN; j++) {
            if (j < POLY_LEN) {
                matrix[i][j] = (shifted_poly >> (POLY_LEN - 1 - j)) & 1;
            } else {
                matrix[i][j] = 0;  // Padded zeroes beyond the polynomial length
            }
        }
    }
    */
}

// Function to calculate the weight distribution
void calculate_weight_distribution(uint64_t matrix[ROWS][N], int *distribution) {
    // uint32_t combination_vector[TOTAL_LEN];
    // uint64_t max_combinations = 1ULL << POLY_LEN;
    //!!!!!!!!!!! not 100% sure but it could be that max_ = 2^32 + 1
    // but the amount of subsets are 2^32-1
    // and we need to start from 1 since empty set is excluded
    uint64_t max_ = 1ULL << 32;   // NUM_SUBSETS;
    printf("%" PRIu64 "\n", max_); 

    uint64_t weight = 0;

    clock_t t0 = clock();

    //Iterate over all possible subsets(excluding the empty set, starting from 1)
    for (size_t subset = 1; subset < max_; subset++) {
        //memset(combination_vector, 0, sizeof(combination_vector));
        uint64_t combination_vector[N] = {0}; 

        // Check each row if it should be included in the current subset
        for (size_t j = 0; j < ROWS; j++) {
            // Check if the ith bit is set in the subset
            if (subset & (1 << j)) {
                for (size_t k = 0; k < N; k++) {
                    // XOR the current combination vector with the ith row of H
                    combination_vector[k] ^= matrix[j][k];
                }
            }
        }

        weight = 0;
        for (size_t m = 0; m < N; m++) {
            weight += combination_vector[m];
        }
        distribution[weight]++;

        if (subset % 1000000 == 1 && subset > 1000) {
            clock_t t1 = clock();
            double time_taken = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
            printf("Time taken: %f\n", time_taken);
            printf("%" PRIu64 "\t/\t", subset, max_);
            printf("%" PRIu64 "\n", max_);
            //printf("Time remaining: %f\n", time_taken * (double)((max_combinations - i) / i));
        }
    }
}

int main() {
    uint64_t H[ROWS][N] = {0};
    int distribution[N + 1] = {0};

    init_crc32_parity_check_matrix(H);
    calculate_weight_distribution(H, distribution);

    printf("Weight Distribution of CRC32 Parity-Check Matrix:\n");
    for (int i = 0; i <= N; i++) {
        if (distribution[i] > 0) {
            printf("Weight %d: %d combinations\n", i, distribution[i]);
        }
    }

    char *fname = "weight.data";
    fclose(fopen(fname, "w"));

    FILE *f = fopen(fname, "w");
    if (f == NULL) {
        printf("Error opening the file %s", fname);
        return -1;
    }

    // print weight distribution
    for (int i = 0; i < N; i++) {
        printf("Weight %d: %u\n", i, distribution[i]);

        fprintf(f, "%u\n", distribution[i]);
    }

    return 0;
}
