#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <string.h>
#include <inttypes.h>
#include <errno.h>
//#define DEBUG 1
#include "mtgp64-fast.h"

void print_double_array(double array[], int size, int block);
void print_uint64_array(uint64_t array[], int size, int block);
void test_init(mtgp64_fast_t mtgp64[], int mexp, int no, int seed, int block);
void test_uint64(mtgp64_fast_t mtgp64[], int mexp, int count, int block);
void test_double(mtgp64_fast_t mtgp64[], int mexp, int count, int block);

void print_double_array(double array[], int size, int block) {
    int b = size / block;

    for (int j = 0; j < 3; j += 3) {
	printf("%.18f %.18f %.18f\n",
	       array[j], array[j + 1], array[j + 2]);
    }
    for (int i = 1; i < block; i++) {
	for (int j = -3; j < 4; j += 3) {
	    printf("%.18f %.18f %.18f\n",
		   array[b * i + j],
		   array[b * i + j + 1],
		   array[b * i + j + 2]);
	}
    }
    for (int j = -3; j < 0; j += 3) {
	printf("%.18f %.18f %.18f\n",
	       array[size + j],
	       array[size + j + 1],
	       array[size + j + 2]);
    }
}

void print_uint64_array(uint64_t array[], int size, int block) {
    int b = size / block;

    for (int j = 0; j < 3; j += 3) {
	printf("%20" PRIu64 " %20" PRIu64 " %20" PRIu64 "\n",
	       array[j], array[j + 1], array[j + 2]);
    }
    for (int i = 1; i < block; i++) {
	for (int j = -3; j < 3; j += 3) {
	    printf("%20" PRIu64 " %20" PRIu64 " %20" PRIu64 "\n",
		   array[b * i + j],
		   array[b * i + j + 1],
		   array[b * i + j + 2]);
	}
    }
    for (int j = -3; j < 0; j += 3) {
	printf("%20" PRIu64 " %20" PRIu64 " %20" PRIu64 "\n",
	       array[size + j],
	       array[size + j + 1],
	       array[size + j + 2]);
    }
}

void test_init(mtgp64_fast_t mtgp64[], int mexp, int no, int seed, int block) {
    int i;
    mtgp64_params_fast_t *params;
    int rc;

    switch (mexp) {
    case 44497:
	params = mtgp64_params_fast_44497;
	break;
    case 23209:
	params = mtgp64_params_fast_23209;
	break;
    default:
	printf("mexp = 23209 or 44497\n");
	return;
    }
    for (i = 0; i < block; i++) {
	rc = mtgp64_init(&mtgp64[i], params + no + i, seed + i);
	if (rc) {
	    printf("failure in mtgp64_init\n");
	    return;
	}
    }
#if defined(DEBUG)
    printf("h_input[0].status[0]:%016"PRIx64"\n", mtgp64[0].status->array[0]);
    printf("h_input[0].status[0]:%016"PRIx64"\n", mtgp64[0].status->array[1]);
    printf("h_input[0].status[0]:%016"PRIx64"\n", mtgp64[0].status->array[2]);
    printf("h_input[0].status[0]:%016"PRIx64"\n", mtgp64[0].status->array[3]);
#endif
}

void test_uint64(mtgp64_fast_t mtgp64[], int mexp, int count, int block) {
    int i, j;
    uint64_t clo;
    uint64_t *array;
    int cuda_large_size;
    int r;
    int size = mexp / 64 + 1;
    int y = size;
    int x;
    float cputime;
    int num;

    for (x = 1; (x != size) && (y > 0); x <<= 1, y >>= 1);
    cuda_large_size = x / 2 * 3;
    r = count % (block * cuda_large_size);
    if (r != 0) {
	count = count + (block * cuda_large_size) - r;
    }
    array = (uint64_t *)malloc(sizeof(uint64_t) * count);
    if (array == NULL) {
	printf("failure in malloc\n");
	return;
    }
    printf("generating 64-bit unsigned random numbers.\n");
    num = count / block;
    clo = clock();
    for (j = 0; j < num; j++) {
	array[j] = mtgp64_genrand_uint64(&mtgp64[0]);
    }
    clo = clock() - clo;
    for (i = 1; i < block; i++) {
	for (j = 0; j < num; j++) {
	    array[i * num + j] = mtgp64_genrand_uint64(&mtgp64[i]);
	}
    }
    cputime = (float)clo * 1000 / CLOCKS_PER_SEC;
    print_uint64_array(array, count, block);
    printf("generated numbers: %d\n", count);
    printf("Processing time: %f (ms)\n", cputime);
    printf("Samples per second: %E \n", num / (cputime * 0.001));
    free(array);
}

void test_double(mtgp64_fast_t mtgp64[], int mexp, int count, int block) {
    int i, j;
    uint64_t clo;
    double *array;
    int cuda_large_size;
    int r;
    int size = mexp / 64 + 1;
    int y = size;
    int x;
    float cputime;
    int num;

    for (x = 1; (x != size) && (y > 0); x <<= 1, y >>= 1);
    cuda_large_size = x / 2 * 3;
    printf("generating 64-bit unsigned random numbers.\n");
    r = count % (block * cuda_large_size);
    if (r != 0) {
	count = count + (block * cuda_large_size) - r;
    }
    array = (double *)malloc(sizeof(double) * count);
    if (array == NULL) {
	printf("failure in malloc\n");
	return;
    }
    clo = clock();
    num = count / block;
    for (j = 0; j < num; j++) {
	array[j] = mtgp64_genrand_close1_open2(&mtgp64[0]);
    }
    clo = clock() - clo;
    for (i = 1; i < block; i++) {
	for (j = 0; j < num; j++) {
	    array[i * num + j] = mtgp64_genrand_close1_open2(&mtgp64[i]);
	}
    }
    cputime = (float)clo * 1000 / CLOCKS_PER_SEC;
    print_double_array(array, count, block);
    printf("generated numbers: %d\n", count);
    printf("Processing time: %f (ms)\n", cputime);
    printf("Samples per second: %E \n", num / (cputime * 0.001));
    free(array);
}

int main(int argc, char *argv[]) {
    int mexp;
    int no;
    int seed;
    int count;
    int block;
    mtgp64_fast_t *mtgp64;
    if (argc < 6) {
	printf("%s mexp no seed count block\n", argv[0]);
	return 1;
    }
    mexp = strtol(argv[1], NULL, 10);
    if (errno) {
	printf("%s mexp no seed count block\n", argv[0]);
	return 1;
    }
    no = strtol(argv[2], NULL, 10);
    if (errno) {
	printf("%s mexp no seed count block\n", argv[0]);
	return 1;
    }
    seed = strtol(argv[3], NULL, 10);
    if (errno) {
	printf("%s mexp no seed count block\n", argv[0]);
	return 1;
    }
    count = strtol(argv[4], NULL, 10);
    if (errno) {
	printf("%s mexp no seed count block\n", argv[0]);
	return 1;
    }
    block = strtol(argv[5], NULL, 10);
    if (errno) {
	printf("%s mexp no seed count block\n", argv[0]);
	return 1;
    }
    mtgp64 = (mtgp64_fast_t *)malloc(sizeof(mtgp64_fast_t) * block);
    if (mtgp64 == NULL) {
	printf("%s alloc error\n", argv[0]);
	return 1;
    }

    test_init(mtgp64, mexp, no, seed, block);
    test_uint64(mtgp64, mexp, count, block);
    test_double(mtgp64, mexp, count, block);

    for (int i = 0; i < block; i++) {
	mtgp64_free(&mtgp64[i]);
    }
    free(mtgp64);
    return 0;
}
