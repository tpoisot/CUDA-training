/**
 * @file mtgp64-ref.c
 *
 * @brief Mersenne Twister for Graphic Processors (MTGP64), which
 * generates 64-bit unsigned integers and double precision floating
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
#include "mtgp64-ref.h"

static uint64_t ini_func1(uint64_t x);
static uint64_t ini_func2(uint64_t x);
static void next_state(mtgp64_ref_t *mtgp64);
static const uint64_t non_zero = 0x4d544750;

/**
 * This function represents a function used in the initialization
 * by init_by_array
 * @param[in] x 64-bit integer
 * @return 64-bit integer
 */
static uint64_t ini_func1(uint64_t x) {
    return (x ^ (x >> 59)) * UINT64_C(2173292883993);
}

/**
 * This function represents a function used in the initialization
 * by init_by_array
 * @param[in] x 64-bit integer
 * @return 64-bit integer
 */
static uint64_t ini_func2(uint64_t x) {
    return (x ^ (x >> 59)) * UINT64_C(58885565329898161);
}

/**
 * The state transition function.
 * @param[in,out] mtgp64 the all in one structure
 */
static void next_state(mtgp64_ref_t *mtgp64) {
    uint64_t *array = mtgp64->status->array;
    int idx;
    int size = mtgp64->status->size;
    uint64_t x;
    uint64_t y;
    uint64_t r;
    uint32_t xh;
    uint32_t xl;
    uint32_t yh;
    uint32_t yl;

    mtgp64->status->idx += 1;
    if (mtgp64->status->idx >= size) {
	mtgp64->status->idx = 0;
    }
    idx = mtgp64->status->idx;
    x = (array[idx] & mtgp64->params.mask) ^ array[(idx + 1) % size];
    y = array[(idx + mtgp64->params.pos) % size];
    xh = (uint32_t)(x >> 32);
    xl = (uint32_t)(x & 0xffffffffU);
    yh = (uint32_t)(y >> 32);
    yl = (uint32_t)(y & 0xffffffffU);
    xh ^= xh << mtgp64->params.sh1;
    xl ^= xl << mtgp64->params.sh1;
    yh = xl ^ (yh >> mtgp64->params.sh2);
    yl = xh ^ (yl >> mtgp64->params.sh2);
    r = ((uint64_t)yh << 32) | yl;
    if (yl & 1) {
	r ^= mtgp64->params.tbl[0];
    }
    if (yl & 2) {
	r ^= mtgp64->params.tbl[1];
    }
    if (yl & 4) {
	r ^= mtgp64->params.tbl[2];
    }
    if (yl & 8) {
	r ^= mtgp64->params.tbl[3];
    }
    array[idx] = r;
}

/**
 * The tempering function.
 * @param[in] tmp_tbl the pre-computed tempering table.
 * @param[in] r the value to be tempered.
 * @param[in] t the tempering helper value.
 * @return the tempered value.
 */
static uint64_t temper(const uint64_t tmp_tbl[4], uint64_t r, uint64_t t) {
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
 * with a 64-bit integer seed. The allocated memory should be freed by
 * calling mtgp64_free(). \b para should be one of the elements in
 * the parameter table (mtgp64-param-ref.c).
 *
 * @param[out] mtgp64 MTGP structure.
 * @param[in] para parameter structure
 * @param[in] seed a 64-bit integer used as the seed.
 * @return memory allocation result. if 0 O.K.
 */
int mtgp64_init(mtgp64_ref_t *mtgp64,
		const mtgp64_params_ref_t *para, uint64_t seed) {
    int i;
    int size = para->mexp / 64 + 1;
    uint64_t hidden_seed;
    uint64_t tmp;
    mtgp64_status_ref_t *st;

    st = (mtgp64_status_ref_t *)malloc(sizeof(mtgp64_status_ref_t)
				     + sizeof(uint64_t) * size);
    if (st == NULL) {
	return -1;
    }
    hidden_seed = para->tbl[2] ^ (para->tbl[3] << 16);
    tmp = hidden_seed >> 32;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(st->array, tmp & 0xff, sizeof(uint64_t) * size);
    mtgp64->params = *para;
    mtgp64->status = st;
    st->size = size;
    st->idx = size - 1;
    st->array[0] = seed;
    st->array[1] = hidden_seed;
    for (i = 1; i < size; i++) {
	st->array[i] ^= UINT64_C(6364136223846793005)
	    * (st->array[i - 1]
	       ^ (st->array[i - 1] >> 62)) + i;
    }
    return 0;
}

/**
 * This function allocates and initializes the internal state array
 * with a 64-bit integer array. The allocated memory should be freed by
 * calling mtgp64_free(). \b para should be one of the elements in
 * the parameter table (mtgp64-param-ref.c).
 *
 * @param[out] mtgp64 MTGP structure.
 * @param[in] para parameter structure
 * @param[in] array a 64-bit integer array used as a seed.
 * @param[in] length length of the array.
 * @return memory allocation result. if 0 O.K.
 */
int mtgp64_init_by_array(mtgp64_ref_t *mtgp64,
			 const mtgp64_params_ref_t *para,
			 const uint64_t *array, int length) {
    int i, j, count;
    uint64_t r;
    int lag;
    int mid;
    int size = para->mexp / 64 + 1;
    uint64_t hidden_seed;
    uint64_t tmp;
    mtgp64_status_ref_t *st;

    st = (mtgp64_status_ref_t *)malloc(sizeof(mtgp64_status_ref_t)
				     + sizeof(uint64_t) * size);
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
    tmp = hidden_seed >> 32;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(st->array, tmp & 0xff, sizeof(uint64_t) * size);
    mtgp64->params = *para;
    mtgp64->status = st;
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
 * calling mtgp64_free(). \b para should be one of the elements in
 * the parameter table (mtgp64-param-ref.c).
 * This is the same algorithm with mtgp64_init_by_array(), but hope to
 * be more useful.
 *
 * @param[out] mtgp64 MTGP structure.
 * @param[in] para parameter structure
 * @param[in] array a character array used as a seed. (terminated by zero.)
 * @return memory allocation result. if 0 O.K.
 */
int mtgp64_init_by_str(mtgp64_ref_t *mtgp64,
		       const mtgp64_params_ref_t *para, const char *array) {
    int i, j, count;
    uint64_t r;
    int lag;
    int mid;
    int size = para->mexp / 64 + 1;
    int length = strlen(array);
    uint64_t hidden_seed;
    uint64_t tmp;
    mtgp64_status_ref_t *st;

    st = (mtgp64_status_ref_t *)malloc(sizeof(mtgp64_status_ref_t)
				     + sizeof(uint64_t) * size);
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
    tmp = hidden_seed >> 32;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(st->array, tmp & 0xff, sizeof(uint64_t) * size);
    mtgp64->params = *para;
    mtgp64->status = st;
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
 * This releases the memory allocated by mtgp64_init(), mtgp64_init_by_array(),
 * mtgp64_init_by_str().
 *
 * @param[in,out] mtgp64 MTGP all in one structure.
 */
void mtgp64_free(mtgp64_ref_t *mtgp64) {
    free(mtgp64->status);
}

/**
 * This function prints the Mersenne exponent and SHA1 of characteristic
 * polynomial of generators state transition function.
 *
 * @param[in] mtgp64 MTGP all in one structure.
 * @param[in,out] fp FILE pointer.
 */
void mtgp64_print_idstring(const mtgp64_ref_t *mtgp64, FILE *fp) {
    int i;
    fprintf(fp, "mtgp64:%d:", mtgp64->params.mexp);
    for (i = 0; i < 20; i++) {
	fprintf(fp, "%02x", (unsigned int)mtgp64->params.poly_sha1[i]);
    }
    fprintf(fp, "\n");
}

/**
 * This function generates and returns 64-bit unsigned integer.
 * mtgp64_init(), mtgp64_init_by_array() or mtgp64_init_by_str() must
 * be called before this function.
 *
 * @param[in,out] mtgp64 MTGP all in one structure.
 * @return 64-bit unsigned integer.
 */
uint64_t mtgp64_genrand_uint64(mtgp64_ref_t *mtgp64) {
    next_state(mtgp64);
    return temper(mtgp64->params.tmp_tbl,
		  mtgp64->status->array[mtgp64->status->idx],
		  mtgp64->status->array[(mtgp64->status->idx
				       + mtgp64->params.pos - 1)
				      % mtgp64->status->size]);
}

/**
 * This function generates and returns double precision pseudorandom
 * number which distributes uniformly in the range [1, 2).
 * mtgp64_init(), mtgp64_init_by_array() or mtgp64_init_by_str() must
 * be called before this function.
 *
 * @param[in,out] mtgp64 MTGP all in one structure.
 * @return double precision floating point pseudorandom number
 */
double mtgp64_genrand_close1_open2(mtgp64_ref_t *mtgp64) {
    union {
	uint64_t u;
	double d;
    } x;
    x.u = mtgp64_genrand_uint64(mtgp64);
    x.u = (x.u >> 12) | UINT64_C(0x3ff0000000000000);
    return x.d;
}

/**
 * This function generates and returns double precision pseudorandom
 * number which distributes uniformly in the range [0, 1).
 * mtgp64_init(), mtgp64_init_by_array() or mtgp64_init_by_str() must
 * be called before this function.
 *
 * @param[in,out] mtgp64 MTGP all in one structure.
 * @return double precision floating point pseudorandom number
 */
double mtgp64_genrand_close_open(mtgp64_ref_t *mtgp64) {
    return mtgp64_genrand_close1_open2(mtgp64) - 1.0;
}

/**
 * This function generates and returns double precision pseudorandom
 * number which distributes uniformly in the range (0, 1].
 * mtgp64_init(), mtgp64_init_by_array() or mtgp64_init_by_str() must
 * be called before this function.
 *
 * @param[in,out] mtgp64 MTGP all in one structure.
 * @return double precision floating point pseudorandom number
 */
double mtgp64_genrand_open_close(mtgp64_ref_t *mtgp64) {
    return 2.0 - mtgp64_genrand_close1_open2(mtgp64);
}

/**
 * This function generates and returns double precision pseudorandom
 * number which distributes uniformly in the range (0, 1).
 * mtgp64_init(), mtgp64_init_by_array() or mtgp64_init_by_str() must
 * be called before this function.
 *
 * @param[in,out] mtgp64 MTGP all in one structure.
 * @return double precision floating point pseudorandom number
 */
double mtgp64_genrand_open_open(mtgp64_ref_t *mtgp64) {
    union {
	uint64_t u;
	double d;
    } x;
    x.u = mtgp64_genrand_uint64(mtgp64);
    x.u = (x.u >> 12) | UINT64_C(0x3ff0000000000001);
    return x.d - 1.0;
}

#if defined(MAIN)
#include <errno.h>
void print_uint64(mtgp64_ref_t *mtgp64, int count);
void print_close1_open2(mtgp64_ref_t *mtgp64, int count);
void print_close_open(mtgp64_ref_t *mtgp64, int count);
void print_open_close(mtgp64_ref_t *mtgp64, int count);
void print_open_open(mtgp64_ref_t *mtgp64, int count);

void print_uint64(mtgp64_ref_t *mtgp64, int count) {
    int i;
    for (i = 0; i < count; i++) {
	printf("%016"PRIx64" ", mtgp64_genrand_uint64(mtgp64));
	if (i % 3 == 2) {
	    printf("\n");
	}
    }
    printf("\n");
}

void print_close1_open2(mtgp64_ref_t *mtgp64, int count) {
    int i;
    for (i = 0; i < count; i++) {
	printf("%.18f ", mtgp64_genrand_close1_open2(mtgp64));
	if (i % 3 == 2) {
	    printf("\n");
	}
    }
    printf("\n");
}

void print_close_open(mtgp64_ref_t *mtgp64, int count) {
    int i;
    for (i = 0; i < count; i++) {
	printf("%.18f ", mtgp64_genrand_close_open(mtgp64));
	if (i % 3 == 2) {
	    printf("\n");
	}
    }
    printf("\n");
}

void print_open_close(mtgp64_ref_t *mtgp64, int count) {
    int i;
    for (i = 0; i < count; i++) {
	printf("%.18f ", mtgp64_genrand_open_close(mtgp64));
	if (i % 3 == 2) {
	    printf("\n");
	}
    }
    printf("\n");
}

void print_open_open(mtgp64_ref_t *mtgp64, int count) {
    int i;
    for (i = 0; i < count; i++) {
	printf("%.18f ", mtgp64_genrand_open_open(mtgp64));
	if (i % 3 == 2) {
	    printf("\n");
	}
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int mexp;
    int no;
    uint64_t seed = 1;
    uint64_t seed_ar[4] = {1, 2, 3, 4};
    char seed_str[] = "\01\02\03\04";
    mtgp64_params_ref_t *params;
    mtgp64_ref_t mtgp64;
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
    case 44497:
	params = mtgp64_params_ref_44497;
	break;
    case 23209:
	params = mtgp64_params_ref_23209;
	break;
    case 110503:
	params = mtgp64_params_ref_110503;
	break;
    default:
	printf("%s: mexp no.\n", argv[0]);
	printf("mexp = 23209, 44497 or 110503 only\n");
	return 4;
    }
    if (no >= 128 || no < 0) {
	printf("%s: mexp no.\n", argv[0]);
	printf("no must be between 0 and 127\n");
	return 5;
    }
    params += no;
    mtgp64_init(&mtgp64, params, seed);
    mtgp64_print_idstring(&mtgp64, stdout);
    printf("init:\n");
    print_uint64(&mtgp64, 1000);
    mtgp64_init_by_array(&mtgp64, params, seed_ar, 4);
    printf("init_array:\n");
    print_uint64(&mtgp64, 1000);
    mtgp64_init_by_str(&mtgp64, params, seed_str);
    printf("init_str:\n");
    print_uint64(&mtgp64, 1000);
    print_close1_open2(&mtgp64, 1000);
    print_close_open(&mtgp64, 1000);
    print_open_close(&mtgp64, 1000);
    print_open_open(&mtgp64, 1000);

    mtgp64_free(&mtgp64);
    return 0;
}
#endif
