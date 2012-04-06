/**
 * @file mtgp32-ref.c
 *
 * @brief Mersenne Twister for Graphic Processors (MTGP32), which
 * generates 32-bit unsigned integers and single precision floating
 * point numbers based on IEEE 754 format.
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (Hiroshima University)
 *
 * Copyright (C) 2009 Mutsuo Saito, Makoto Matsumoto and
 * Hiroshima University. All rights reserved.
 *
 * The new BSD License is applied to this software, see LICENSE.txt
 */
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "mtgp32-ref.h"

static uint32_t ini_func1(uint32_t x);
static uint32_t ini_func2(uint32_t x);
static void next_state(mtgp32_ref_t *mtgp32);
static const uint32_t non_zero = 0x4d544750;

/**
 * This function represents a function used in the initialization
 * by init_by_array
 * @param[in] x 32-bit integer
 * @return 32-bit integer
 */
static uint32_t ini_func1(uint32_t x) {
    return (x ^ (x >> 27)) * UINT32_C(1664525);
}

/**
 * This function represents a function used in the initialization
 * by init_by_array
 * @param[in] x 32-bit integer
 * @return 32-bit integer
 */
static uint32_t ini_func2(uint32_t x) {
    return (x ^ (x >> 27)) * UINT32_C(1566083941);
}

/**
 * The state transition function.
 * @param[in,out] mtgp32 the all in one structure
 */
static void next_state(mtgp32_ref_t *mtgp32) {
    uint32_t *array = mtgp32->status->array;
    int idx;
    int size = mtgp32->status->size;
    uint32_t x;
    uint32_t y;
    uint32_t yl;

    mtgp32->status->idx += 1;
    if (mtgp32->status->idx >= size) {
	mtgp32->status->idx = 0;
    }
    idx = mtgp32->status->idx;
    x = (array[idx] & mtgp32->params.mask) ^ array[(idx + 1) % size];
    y = array[(idx + mtgp32->params.pos) % size];
    x ^= x << mtgp32->params.sh1;
    y = x ^ (y >> mtgp32->params.sh2);
    yl = y & 0x0f;
    if (yl & 1) {
	y ^= mtgp32->params.tbl[0];
    }
    if (yl & 2) {
	y ^= mtgp32->params.tbl[1];
    }
    if (yl & 4) {
	y ^= mtgp32->params.tbl[2];
    }
    if (yl & 8) {
	y ^= mtgp32->params.tbl[3];
    }
    array[idx] = y;
}

/**
 * The tempering function.
 * @param[in] tmp_tbl the pre-computed tempering table.
 * @param[in] r the value to be tempered.
 * @param[in] t the tempering helper value.
 * @return the tempered value.
 */
static uint32_t temper(const uint32_t tmp_tbl[4], uint32_t r, uint32_t t) {
    t ^= t >> 16;
    t ^= t >> 8;
    if (t & 1) {
	r ^= tmp_tbl[0];
    }
    if (t & 2) {
	r ^= tmp_tbl[1];
    }
    if (t & 4) {
	r ^= tmp_tbl[2];
    }
    if (t & 8) {
	r ^= tmp_tbl[3];
    }
    return r;
}

/*----------------
  PUBLIC FUNCTIONS
  ----------------*/
/**
 * This function allocates and initializes the internal state array
 * with a 32-bit integer seed. The allocated memory should be freed by
 * calling mtgp32_free(). \b para should be one of the elements in
 * the parameter table (mtgp32-param-ref.c).
 *
 * @param[out] mtgp32 MTGP structure.
 * @param[in] para parameter structure
 * @param[in] seed a 32-bit integer used as the seed.
 * @return memory allocation result. if 0 O.K.
 */
int mtgp32_init(mtgp32_ref_t *mtgp32,
		const mtgp32_params_ref_t *para, uint32_t seed) {
    int i;
    int size = para->mexp / 32 + 1;
    uint32_t hidden_seed;
    uint32_t tmp;
    mtgp32_status_ref_t *st;

    st = (mtgp32_status_ref_t *)malloc(sizeof(mtgp32_status_ref_t)
				     + sizeof(uint32_t) * size);
    if (st == NULL) {
	return -1;
    }
    hidden_seed = para->tbl[2] ^ (para->tbl[3] << 16);
    tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(st->array, tmp & 0xff, sizeof(uint32_t) * size);
    mtgp32->params = *para;
    mtgp32->status = st;
    st->size = size;
    st->idx = size - 1;
    st->array[0] = seed;
    st->array[1] = hidden_seed;
    for (i = 1; i < size; i++) {
	st->array[i] ^= UINT32_C(1812433253) * (st->array[i - 1]
						^ (st->array[i - 1] >> 30))
	    + i;
    }
    return 0;
}

/**
 * This function allocates and initializes the internal state array
 * with a 32-bit integer array. The allocated memory should be freed by
 * calling mtgp32_free(). \b para should be one of the elements in
 * the parameter table (mtgp32-param-ref.c).
 *
 * @param[out] mtgp32 MTGP structure.
 * @param[in] para parameter structure
 * @param[in] array a 32-bit integer array used as a seed.
 * @param[in] length length of the array.
 * @return memory allocation result. if 0 O.K.
 */
int mtgp32_init_by_array(mtgp32_ref_t *mtgp32,
			 const mtgp32_params_ref_t *para,
			 const uint32_t *array, int length) {
    int i, j, count;
    uint32_t r;
    int lag;
    int mid;
    int size = para->mexp / 32 + 1;
    uint32_t hidden_seed;
    uint32_t tmp;
    mtgp32_status_ref_t *st;

    st = (mtgp32_status_ref_t *)malloc(sizeof(mtgp32_status_ref_t)
				     + sizeof(uint32_t) * size);
    if (st == NULL) {
	return -1;
    }
    if (size >= 623) {
	lag = 11;
    } else if (size >= 68) {
	lag = 7;
    } else if (size >= 39) {
	lag = 5;
    } else {
	lag = 3;
    }
    mid = (size - lag) / 2;

    hidden_seed = para->tbl[2] ^ (para->tbl[3] << 16);
    tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(st->array, tmp & 0xff, sizeof(uint32_t) * size);
    mtgp32->params = *para;
    mtgp32->status = st;
    st->size = size;
    st->idx = size - 1;
    st->array[0] = hidden_seed;

    if (length + 1 > size) {
	count = length + 1;
    } else {
	count = size;
    }
    r = ini_func1(st->array[0] ^ st->array[mid] ^ st->array[size - 1]);
    st->array[mid] += r;
    r += length;
    st->array[(mid + lag) % size] += r;
    st->array[0] = r;
    i = 1;
    count--;
    for (i = 1, j = 0; (j < count) && (j < length); j++) {
	r = ini_func1(st->array[i] ^ st->array[(i + mid) % size]
		      ^ st->array[(i + size - 1) % size]);
	st->array[(i + mid) % size] += r;
	r += array[j] + i;
	st->array[(i + mid + lag) % size] += r;
	st->array[i] = r;
	i = (i + 1) % size;
    }
    for (; j < count; j++) {
	r = ini_func1(st->array[i] ^ st->array[(i + mid) % size]
		      ^ st->array[(i + size - 1) % size]);
	st->array[(i + mid) % size] += r;
	r += i;
	st->array[(i + mid + lag) % size] += r;
	st->array[i] = r;
	i = (i + 1) % size;
    }
    for (j = 0; j < size; j++) {
	r = ini_func2(st->array[i] + st->array[(i + mid) % size]
		      + st->array[(i + size - 1) % size]);
	st->array[(i + mid) % size] ^= r;
	r -= i;
	st->array[(i + mid + lag) % size] ^= r;
	st->array[i] = r;
	i = (i + 1) % size;
    }
    if (st->array[size - 1] == 0) {
	st->array[size - 1] = non_zero;
    }
    return 0;
}

/**
 * This function allocates and initializes the internal state array
 * with a character array. The allocated memory should be freed by
 * calling mtgp32_free(). \b para should be one of the elements in
 * the parameter table (mtgp32-param-ref.c).
 * This is the same algorithm with mtgp32_init_by_array(), but hope to
 * be more useful.
 *
 * @param[out] mtgp32 MTGP structure.
 * @param[in] para parameter structure
 * @param[in] array a character array used as a seed. (terminated by zero.)
 * @return memory allocation result. if 0 O.K.
 */
int mtgp32_init_by_str(mtgp32_ref_t *mtgp32,
		       const mtgp32_params_ref_t *para, const char *array) {
    int i, j, count;
    uint32_t r;
    int lag;
    int mid;
    int size = para->mexp / 32 + 1;
    int length = strlen(array);
    uint32_t hidden_seed;
    uint32_t tmp;
    mtgp32_status_ref_t *st;

    st = (mtgp32_status_ref_t *)malloc(sizeof(mtgp32_status_ref_t)
				     + sizeof(uint32_t) * size);
    if (st == NULL) {
	return -1;
    }
    if (size >= 623) {
	lag = 11;
    } else if (size >= 68) {
	lag = 7;
    } else if (size >= 39) {
	lag = 5;
    } else {
	lag = 3;
    }
    mid = (size - lag) / 2;

    hidden_seed = para->tbl[2] ^ (para->tbl[3] << 16);
    tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(st->array, tmp & 0xff, sizeof(uint32_t) * size);
    mtgp32->params = *para;
    mtgp32->status = st;
    st->size = size;
    st->idx = size - 1;
    st->array[0] = hidden_seed;

    if (length + 1 > size) {
	count = length + 1;
    } else {
	count = size;
    }
    r = ini_func1(st->array[0] ^ st->array[mid] ^ st->array[size - 1]);
    st->array[mid] += r;
    r += length;
    st->array[(mid + lag) % size] += r;
    st->array[0] = r;
    i = 1;
    count--;
    for (i = 1, j = 0; (j < count) && (j < length); j++) {
	r = ini_func1(st->array[i] ^ st->array[(i + mid) % size]
		      ^ st->array[(i + size - 1) % size]);
	st->array[(i + mid) % size] += r;
	r += array[j] + i;
	st->array[(i + mid + lag) % size] += r;
	st->array[i] = r;
	i = (i + 1) % size;
    }
    for (; j < count; j++) {
	r = ini_func1(st->array[i] ^ st->array[(i + mid) % size]
		      ^ st->array[(i + size - 1) % size]);
	st->array[(i + mid) % size] += r;
	r += i;
	st->array[(i + mid + lag) % size] += r;
	st->array[i] = r;
	i = (i + 1) % size;
    }
    for (j = 0; j < size; j++) {
	r = ini_func2(st->array[i] + st->array[(i + mid) % size]
		      + st->array[(i + size - 1) % size]);
	st->array[(i + mid) % size] ^= r;
	r -= i;
	st->array[(i + mid + lag) % size] ^= r;
	st->array[i] = r;
	i = (i + 1) % size;
    }
    if (st->array[size - 1] == 0) {
	st->array[size - 1] = non_zero;
    }
    return 0;
}

/**
 * This releases the memory allocated by mtgp32_init(), mtgp32_init_by_array(),
 * mtgp32_init_by_str().
 *
 * @param[in,out] mtgp32 MTGP all in one structure.
 */
void mtgp32_free(mtgp32_ref_t *mtgp32) {
    free(mtgp32->status);
}

/**
 * This function prints the Mersenne exponent and SHA1 of characteristic
 * polynomial of generators state transition function.
 *
 * @param[in] mtgp32 MTGP all in one structure.
 * @param[in,out] fp FILE pointer.
 */
void mtgp32_print_idstring(const mtgp32_ref_t *mtgp32, FILE *fp) {
    int i;
    fprintf(fp, "mtgp32:%d:", mtgp32->params.mexp);
    for (i = 0; i < 20; i++) {
	fprintf(fp, "%02x", (unsigned int)mtgp32->params.poly_sha1[i]);
    }
    fprintf(fp, "\n");
}

/**
 * This function generates and returns 32-bit unsigned integer.
 * mtgp32_init(), mtgp32_init_by_array() or mtgp32_init_by_str() must
 * be called before this function.
 *
 * @param[in,out] mtgp32 MTGP all in one structure.
 * @return 32-bit unsigned integer.
 */
uint32_t mtgp32_genrand_uint32(mtgp32_ref_t *mtgp32) {
    next_state(mtgp32);
    return temper(mtgp32->params.tmp_tbl,
		  mtgp32->status->array[mtgp32->status->idx],
		  mtgp32->status->array[(mtgp32->status->idx
					 + mtgp32->params.pos - 1)
					% mtgp32->status->size]);
}

/**
 * This function generates and returns single precision pseudorandom
 * number which distributes uniformly in the range [1, 2).
 * mtgp32_init(), mtgp32_init_by_array() or mtgp32_init_by_str() must
 * be called before this function.
 *
 * @param[in,out] mtgp32 MTGP all in one structure.
 * @return single precision floating point pseudorandom number
 */
float mtgp32_genrand_close1_open2(mtgp32_ref_t *mtgp32) {
    union {
	uint32_t u;
	float f;
    } x;
    x.u = mtgp32_genrand_uint32(mtgp32);
    x.u = (x.u >> 9) | UINT32_C(0x3f800000);
    return x.f;
}

/**
 * This function generates and returns single precision pseudorandom
 * number which distributes uniformly in the range [0, 1).
 * mtgp32_init(), mtgp32_init_by_array() or mtgp32_init_by_str() must
 * be called before this function.
 *
 * @param[in,out] mtgp32 MTGP all in one structure.
 * @return single precision floating point pseudorandom number
 */
float mtgp32_genrand_close_open(mtgp32_ref_t *mtgp32) {
    return mtgp32_genrand_close1_open2(mtgp32) - 1.0F;
}

/**
 * This function generates and returns single precision pseudorandom
 * number which distributes uniformly in the range (0, 1].
 * mtgp32_init(), mtgp32_init_by_array() or mtgp32_init_by_str() must
 * be called before this function.
 *
 * @param[in,out] mtgp32 MTGP all in one structure.
 * @return single precision floating point pseudorandom number
 */
float mtgp32_genrand_open_close(mtgp32_ref_t *mtgp32) {
    return 2.0F - mtgp32_genrand_close1_open2(mtgp32);
}

/**
 * This function generates and returns single precision pseudorandom
 * number which distributes uniformly in the range (0, 1).
 * mtgp32_init(), mtgp32_init_by_array() or mtgp32_init_by_str() must
 * be called before this function.
 *
 * @param[in,out] mtgp32 MTGP all in one structure.
 * @return single precision floating point pseudorandom number
 */
float mtgp32_genrand_open_open(mtgp32_ref_t *mtgp32) {
    union {
	uint32_t u;
	float f;
    } x;
    x.u = mtgp32_genrand_uint32(mtgp32);
    x.u = (x.u >> 9) | UINT32_C(0x3f800001);
    return x.f - 1.0F;
}

#if defined(MAIN)
#include <errno.h>
void print_uint32(mtgp32_ref_t *mtgp32, int count);
void print_close1_open2(mtgp32_ref_t *mtgp32, int count);
void print_close_open(mtgp32_ref_t *mtgp32, int count);
void print_open_close(mtgp32_ref_t *mtgp32, int count);
void print_open_open(mtgp32_ref_t *mtgp32, int count);

void print_uint32(mtgp32_ref_t *mtgp32, int count) {
    int i;
    for (i = 0; i < count; i++) {
	printf("%10"PRIu32" ", mtgp32_genrand_uint32(mtgp32));
	if (i % 5 == 4) {
	    printf("\n");
	}
    }
    if (i % 5 != 0) {
	printf("\n");
    }
}

void print_close1_open2(mtgp32_ref_t *mtgp32, int count) {
    int i;
    for (i = 0; i < count; i++) {
	printf("%.8f ", mtgp32_genrand_close1_open2(mtgp32));
	if (i % 5 == 4) {
	    printf("\n");
	}
    }
    if (i % 5 != 0) {
	printf("\n");
    }
    printf("\n");
}

void print_close_open(mtgp32_ref_t *mtgp32, int count) {
    int i;
    for (i = 0; i < count; i++) {
	printf("%.8f ", mtgp32_genrand_close_open(mtgp32));
	if (i % 5 == 4) {
	    printf("\n");
	}
    }
    if (i % 5 != 0) {
	printf("\n");
    }
}

void print_open_close(mtgp32_ref_t *mtgp32, int count) {
    int i;
    for (i = 0; i < count; i++) {
	printf("%.8f ", mtgp32_genrand_open_close(mtgp32));
	if (i % 5 == 4) {
	    printf("\n");
	}
    }
    if (i % 5 != 0) {
	printf("\n");
    }
}

void print_open_open(mtgp32_ref_t *mtgp32, int count) {
    int i;
    for (i = 0; i < count; i++) {
	printf("%.8f ", mtgp32_genrand_open_open(mtgp32));
	if (i % 5 == 4) {
	    printf("\n");
	}
    }
    if (i % 5 != 0) {
	printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int mexp;
    int no;
    uint32_t seed = 1;
    uint32_t seed_ar[4] = {1, 2, 3, 4};
    char seed_str[] = "\01\02\03\04";
    mtgp32_params_ref_t *params;
    mtgp32_ref_t mtgp32;
    if (argc <= 2) {
	printf("%s: mexp no.\n", argv[0]);
	return 1;
    }
    mexp = strtol(argv[1], NULL, 10);
    if (errno) {
	printf("%s: mexp no.\n", argv[0]);
	return 2;
    }
    no = strtol(argv[2], NULL, 10);
    if (errno) {
	printf("%s: mexp no.\n", argv[0]);
	return 3;
    }
    switch (mexp) {
    case 11213:
	params = mtgp32_params_ref_11213;
	break;
    case 23209:
	params = mtgp32_params_ref_23209;
	break;
    case 44497:
	params = mtgp32_params_ref_44497;
	break;
    default:
	printf("%s: mexp no.\n", argv[0]);
	printf("mexp should be 11213, 23209 or 44497 only\n");
	return 4;
    }
    if (no >= 128 || no < 0) {
	printf("%s: mexp no.\n", argv[0]);
	printf("no must be between 0 and 127\n");
	return 5;
    }
    params += no;
    mtgp32_init(&mtgp32, params, seed);
    mtgp32_print_idstring(&mtgp32, stdout);
    printf("init:\n");
    print_uint32(&mtgp32, 1000);
    mtgp32_init_by_array(&mtgp32, params, seed_ar, 4);
    printf("init_array:\n");
    print_uint32(&mtgp32, 1000);
    mtgp32_init_by_str(&mtgp32, params, seed_str);
    printf("init_str:\n");
    print_uint32(&mtgp32, 1000);
    print_close1_open2(&mtgp32, 1000);
    print_close_open(&mtgp32, 1000);
    print_open_close(&mtgp32, 1000);
    print_open_open(&mtgp32, 1000);

    mtgp32_free(&mtgp32);
    return 0;
}
#endif
