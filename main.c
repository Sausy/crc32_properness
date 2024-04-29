#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>
#include <float.h>

// local includes
#include "CRC32_LUT_generator.h"
#include "Analyse_weight_distribution.h"


long double buffer_calc(long double Pe, int32_t n, int32_t i, uint64_t Ai){
    long double P0 = powl(Pe, (long double)i);
    long double P1 = 1 - Pe;
    long double powerP1 = (long double)n - (long double)i;
    long double P2 = powl(P1, powerP1);

    long double ret = ((long double)Ai * P0 * P2);

    //printf("---n-i: %lf\n", (double)(powerP1));
    //printf("P0: %G\n", (double)P0);
    //printf("P2: %G\n", (double)P0);
    //printf("ret: %G\n", (double)ret);
    return ret; 
}

int main(void){
    // run all tests
    test_all_crc();

    return 0; 

    // to store data to a file
    char *fname = "weight.data";
    fclose(fopen(fname, "w"));

    uint32_t crcTable[CRC_TABLE_SIZE];
    uint32_t polynomial = 0xF1922815;
    printf("Polynomial: 0x%X\n", polynomial);

    generate_crc32_table(crcTable, polynomial);
    print_CRC32_table(crcTable);

    /*

    uint64_t weight_distribution[CODEWORD_LENGTH] = {0};
    get_weight_distribution(weight_distribution, crcTable);

    

    FILE *f = fopen(fname, "w");
    if (f == NULL) {
        printf("Error opening the file %s", fname);
        return -1;
    }

    // print weight distribution
    for (int i = 0; i < CODEWORD_LENGTH; i++){
        printf("Weight %d: %"PRIu64"\n", i, weight_distribution[i]);

        fprintf(f, "%"PRIu64"\n", weight_distribution[i]);
    }

    fclose(f);

    int32_t n = CODEWORD_LENGTH;
    printf("n: %d\n", n);
    long double Rcrc = 0.0;
    long double Pe = 0.01;  // 10^-2 according to norm

    for (int32_t i = 0; i < n; i++) {
        Rcrc += buffer_calc(Pe, n, i, weight_distribution[i]);
    }

    printf("Rcrc: %e\n", (double)Rcrc);
    printf("Rcrc: %G\n", (double)Rcrc);
    */

    return 0;
}