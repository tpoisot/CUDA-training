#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <string.h>
#include <inttypes.h>
#include <errno.h>
//#define DEBUG 1
#include "mtgp32-fast.h"

void print_float_array(float array[], int size, int block);
void print_uint32_array(uint32_t array[], int size, int block);
void test_init(mtgp32_fast_t mtgp32[], int mexp, int no, int seed, int block);
void test_uint32(mtgp32_fast_t mtgp32[], int mexp, int count, int block);
void test_float(mtgp32_fast_t mtgp32[], int mexp, int count, int block);

void print_float_array(float array[], int size, int block) {
    int b = size / block;

    for (int j = 0; j < 5; j += 5) {
	printf("%.10f %.10f %.10f %.10f %.10f\n",
	       array[j], array[j + 1], array[j + 2],
	       array[j + 3], array[j + 4]);
    }
    for (int i = 1; i < block; i++) {
	for (int j = -5; j < 5; j += 5) {
	    printf("%.10f %.10f %.10f %.10f %.10f\n",
		   array[b * i + j],
		   array[b * i + j + 1],
		   array[b * i + j + 2],
		   array[b * i + j + 3],
		   array[b * i + j + 4]);
	}
    }
    for (int j = -5; j < 0; j += 5) {
	printf("%.10f %.10f %.10f %.10f %.10f\n",
	       array[size + j],
	       array[size + j + 1],
	       array[size + j + 2],
	       array[size + j + 3],
	       array[size + j + 4]);
    }
}

void print_uint32_array(uint32_t array[], int size, int block) {
    int b = size / block;

    for (int j = 0; j < 5; j += 5) {
	printf("%10" PRIu32 " %10" PRIu32 " %10" PRIu32
	       " %10" PRIu32 " %10" PRIu32 "\n",
	       array[j], array[j + 1], array[j + 2],
	       array[j + 3], array[j + 4]);
    }
    for (int i = 1; i < block; i++) {
	for (int j = -5; j < 5; j += 5) {
	    printf("%10" PRIu32 " %10" PRIu32 " %10" PRIu32
		   " %10" PRIu32 " %10" PRIu32 "\n",
		   array[b * i + j],
		   array[b * i + j + 1],
		   array[b * i + j + 2],
		   array[b * i + j + 3],
		   array[b * i + j + 4]);
	}
    }
    for (int j = -5; j < 0; j += 5) {
	printf("%10" PRIu32 " %10" PRIu32 " %10" PRIu32
	       " %10" PRIu32 " %10" PRIu32 "\n",
	       array[size + j],
	       array[size + j + 1],
	       array[size + j + 2],
	       array[size + j + 3],
	       array[size + j + 4]);
    }
}

void test_init(mtgp32_fast_t mtgp32[], int mexp, int no, int seed, int block) {
    int i;
    mtgp32_params_fast_t *params;
    int rc;

    switch (mexp) {
    case 44497:
	params = mtgp32_params_fast_44497;
	break;
    case 23209:
	params = mtgp32_params_fast_23209;
	break;
    default:
	printf("mexp = 23209 or 44497\n");
	return;
    }
    for (i = 0; i < block; i++) {
	rc = mtgp32_init(&mtgp32[i], params + no + i, seed + i);
	if (rc) {
	    printf("failure in mtgp32_init\n");
	    return;
	}
    }
#if defined(DEBUG)
    printf("h_input[0].status[0]:%08"PRIx32"\n", mtgp32[0].status->array[0]);
    printf("h_input[0].status[1]:%08"PRIx32"\n", mtgp32[0].status->array[1]);
    printf("h_input[0].status[2]:%08"PRIx32"\n", mtgp32[0].status->array[2]);
    printf("h_input[0].status[3]:%08"PRIx32"\n", mtgp32[0].status->array[3]);
#endif
}

void test_uint32(mtgp32_fast_t mtgp32[], int mexp, int count, int block) {
    int i, j;
    uint32_t clo;
    uint32_t *array;
    int cuda_large_size;
    int r;
    int size = mexp / 32 + 1;
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
    array = (uint32_t *)malloc(sizeof(uint32_t) * count);
    if (array == NULL) {
	printf("failure in malloc\n");
	return;
    }
    printf("generating 32-bit unsigned random numbers.\n");
    num = count / block;
#ifdef DEBUG
    printf("LARGE_SIZE:%d\n", cuda_large_size);
    printf("count:%d\n", count);
    printf("num:%d\n", num);
#endif
    clo = clock();
    for (j = 0; j < num; j++) {
	array[j] = mtgp32_genrand_uint32(&mtgp32[0]);
    }
    clo = clock() - clo;
    for (i = 1; i < block; i++) {
	for (j = 0; j < num; j++) {
	    array[i * num + j] = mtgp32_genrand_uint32(&mtgp32[i]);
	}
    }
    cputime = (float)clo * 1000 / CLOCKS_PER_SEC;
    print_uint32_array(array, count, block);
    printf("generated numbers: %d\n", count);
    printf("Processing time: %f (ms)\n", cputime);
    printf("Samples per second: %E \n", num / (cputime * 0.001));
    free(array);
}

void test_float(mtgp32_fast_t mtgp32[], int mexp, int count, int block) {
    int i, j;
    uint32_t clo;
    float *array;
    int cuda_large_size;
    int r;
    int size = mexp / 32 + 1;
    int y = size;
    int x;
    float cputime;
    int num;

    for (x = 1; (x != size) && (y > 0); x <<= 1, y >>= 1);
    cuda_large_size = x / 2 * 3;
    printf("generating 32-bit unsigned random numbers.\n");
    r = count % (block * cuda_large_size);
    if (r != 0) {
	count = count + (block * cuda_large_size) - r;
    }
    array = (float *)malloc(sizeof(float) * count);
    if (array == NULL) {
	printf("failure in malloc\n");
	return;
    }
    clo = clock();
    num = count / block;
    for (j = 0; j < num; j++) {
	array[j] = mtgp32_genrand_close1_open2(&mtgp32[0]);
    }
    clo = clock() - clo;
    for (i = 1; i < block; i++) {
	for (j = 0; j < num; j++) {
	    array[i * num + j] = mtgp32_genrand_close1_open2(&mtgp32[i]);
	}
    }
    cputime = (float)clo * 1000 / CLOCKS_PER_SEC;
    print_float_array(array, count, block);
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
    mtgp32_fast_t *mtgp32;
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
    mtgp32 = (mtgp32_fast_t *)malloc(sizeof(mtgp32_fast_t) * block);
    if (mtgp32 == NULL) {
	printf("%s alloc error\n", argv[0]);
	return 1;
    }

    test_init(mtgp32, mexp, no, seed, block);
    test_uint32(mtgp32, mexp, count, block);
    test_float(mtgp32, mexp, count, block);

    for (int i = 0; i < block; i++) {
	mtgp32_free(&mtgp32[i]);
    }
    free(mtgp32);
    return 0;
}
