/**
 * @file mtgp64-fast.c
 *
 * @brief Mersenne Twister for Graphic Processors (mtgp64), which
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
#include "mtgp64-fast.h"

static uint64_t ini_func1(uint64_t x);
static uint64_t ini_func2(uint64_t x);
static int alloc_state(mtgp64_fast_t *mtgp64, const mtgp64_params_fast_t *para);
static const uint64_t non_zero = 0x4d544750;

/**
 * \private
 * This function allocates the internal state vector.
 *
 * @param[in,out] mtgp64 MTGP all in one structure.
 * @param[in] para a parameter structure.
 * @return allocation status. if 0 O.K.
 */
static int alloc_state(mtgp64_fast_t *mtgp64,
		       const mtgp64_params_fast_t *para) {
    int size = para->mexp / 64 + 1;
    int large_size;
    mtgp64_status_fast_t *st;

    int x;
    int y = size;
    for (x = 1; (x != size) && (y > 0); x <<= 1, y >>= 1);
    large_size = x;

    st = (mtgp64_status_fast_t *)malloc(sizeof(mtgp64_status_fast_t)
				     + sizeof(uint64_t) * large_size);
    if (st == NULL) {
	return -1;
    }
    mtgp64->params = *para;
    mtgp64->status = st;
    st->size = size;
    st->large_size = large_size;
    st->large_mask = large_size -1;
    return 0;
}

/**
 * \private
 * This function represents a function used in the initialization
 * by mtgp64_init_by_array() and mtgp64_init_by_str().
 *
 * @param[in] x 64-bit integer
 * @return 64-bit integer
 */
static uint64_t ini_func1(uint64_t x) {
    return (x ^ (x >> 59)) * UINT64_C(2173292883993);
}

/**
 * \private
 * This function represents a function used in the initialization
 * by mtgp64_init_by_array() and mtgp64_init_by_str().
 *
 * @param[in] x 64-bit integer
 * @return 64-bit integer
 */
static uint64_t ini_func2(uint64_t x) {
    return (x ^ (x >> 59)) * UINT64_C(58885565329898161);
}

/*----------------
  PUBLIC FUNCTIONS
  ----------------*/
/**
 * \public
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
int mtgp64_init(mtgp64_fast_t *mtgp64,
		const mtgp64_params_fast_t *para, uint64_t seed) {
    int rc;
    rc = alloc_state(mtgp64, para);
    if (rc) {
	return rc;
    }
    mtgp64->status->idx = mtgp64->status->size - 1;
    mtgp64_init_state(mtgp64->status->array, para, seed);
    return 0;
}

/**
 * This function initializes the internal state array with a 64-bit
 * integer seed. The allocated memory should be freed by calling
 * mtgp64_free(). \b para should be one of the elements in the
 * parameter table (mtgp64-param-ref.c).
 *
 * This function is call by cuda program, because cuda program uses
 * another structure and another allocation method.
 *
 * @param[out] array MTGP internal status vector.
 * @param[in] para parameter structure
 * @param[in] seed a 64-bit integer used as the seed.
 */
void mtgp64_init_state(uint64_t array[],
		      const mtgp64_params_fast_t *para, uint64_t seed) {
    int i;
    int size = para->mexp / 64 + 1;
    uint64_t hidden_seed;
    uint64_t tmp;
    hidden_seed = para->tbl[4] ^ (para->tbl[8] << 16);
    tmp = hidden_seed >> 32;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(array, tmp & 0xff, sizeof(uint64_t) * size);
    array[0] = seed;
    array[1] = hidden_seed;
    for (i = 1; i < size; i++) {
	array[i] ^= UINT64_C(6364136223846793005)
	    * (array[i - 1]
	       ^ (array[i - 1] >> 62)) + i;
    }
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
int mtgp64_init_by_array(mtgp64_fast_t *mtgp64,
			 const mtgp64_params_fast_t *para,
			 uint64_t *array, int length) {
    int i, j, count;
    uint64_t r;
    int lag;
    int mid;
    int size = para->mexp / 64 + 1;
    uint64_t hidden_seed;
    uint64_t tmp;
    mtgp64_status_fast_t *st;
    int rc;

    rc = alloc_state(mtgp64, para);
    if (rc) {
	return rc;
    }

    st = mtgp64->status;
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

    hidden_seed = para->tbl[4] ^ (para->tbl[8] << 16);
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
 * @return memory allocation result. if 0 then O.K.
 */
int mtgp64_init_by_str(mtgp64_fast_t *mtgp64,
		       const mtgp64_params_fast_t *para, char *array) {
    int i, j, count;
    uint64_t r;
    int lag;
    int mid;
    int size = para->mexp / 64 + 1;
    int length = strlen(array);
    uint64_t hidden_seed;
    uint64_t tmp;
    mtgp64_status_fast_t *st;
    int rc;

    rc = alloc_state(mtgp64, para);
    if (rc) {
	return rc;
    }

    st = mtgp64->status;
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

    hidden_seed = para->tbl[4] ^ (para->tbl[8] << 16);
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
void mtgp64_free(mtgp64_fast_t *mtgp64) {
    free(mtgp64->status);
}

/**
 * This function prints the Mersenne exponent and SHA1 of characteristic
 * polynomial of generators state transition function.
 *
 * @param[in] mtgp64 MTGP all in one structure.
 * @param[in,out] fp FILE pointer.
 */
void mtgp64_print_idstring(const mtgp64_fast_t *mtgp64, FILE *fp) {
    int i;
    fprintf(fp, "mtgp64:%d:", mtgp64->params.mexp);
    for (i = 0; i < 20; i++) {
	fprintf(fp, "%02x", (unsigned int)mtgp64->params.poly_sha1[i]);
    }
    fprintf(fp, "\n");
}

#if defined(MAIN)
#include <errno.h>
void print_uint64(mtgp64_fast_t *mtgp64, int count);
void print_close1_open2(mtgp64_fast_t *mtgp64, int count);
void print_close_open(mtgp64_fast_t *mtgp64, int count);
void print_open_close(mtgp64_fast_t *mtgp64, int count);
void print_open_open(mtgp64_fast_t *mtgp64, int count);

void print_uint64(mtgp64_fast_t *mtgp64, int count) {
    int i;
    for (i = 0; i < count; i++) {
	printf("%016"PRIx64" ", mtgp64_genrand_uint64(mtgp64));
	if (i % 3 == 2) {
	    printf("\n");
	}
    }
    printf("\n");
}

void print_close1_open2(mtgp64_fast_t *mtgp64, int count) {
    int i;
    for (i = 0; i < count; i++) {
	printf("%.18f ", mtgp64_genrand_close1_open2(mtgp64));
	if (i % 3 == 2) {
	    printf("\n");
	}
    }
    printf("\n");
}

void print_close_open(mtgp64_fast_t *mtgp64, int count) {
    int i;
    for (i = 0; i < count; i++) {
	printf("%.18f ", mtgp64_genrand_close_open(mtgp64));
	if (i % 3 == 2) {
	    printf("\n");
	}
    }
    printf("\n");
}

void print_open_close(mtgp64_fast_t *mtgp64, int count) {
    int i;
    for (i = 0; i < count; i++) {
	printf("%.18f ", mtgp64_genrand_open_close(mtgp64));
	if (i % 3 == 2) {
	    printf("\n");
	}
    }
    printf("\n");
}

void print_open_open(mtgp64_fast_t *mtgp64, int count) {
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
    mtgp64_params_fast_t *params;
    mtgp64_fast_t mtgp64;
    int rc;

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
	params = mtgp64_params_fast_44497;
	break;
    case 23209:
	params = mtgp64_params_fast_23209;
	break;
    case 110503:
	params = mtgp64_params_fast_110503;
	break;
    default:
	printf("%s: mexp no.\n", argv[0]);
	printf("mexp = 23209, 44497 or 110503\n");
	return 4;
    }
    if (no >= 128 || no < 0) {
	printf("%s: mexp no.\n", argv[0]);
	printf("no must be between 0 and 127\n");
	return 5;
    }
    params += no;
    rc = mtgp64_init(&mtgp64, params, seed);
    if (rc) {
	printf("failure in mtgp64_init\n");
	return -1;
    }
    mtgp64_print_idstring(&mtgp64, stdout);
    printf("init:\n");
    print_uint64(&mtgp64, 1000);
    mtgp64_free(&mtgp64);

    rc = mtgp64_init_by_array(&mtgp64, params, seed_ar, 4);
    if (rc) {
	printf("failure in mtgp64_init_by_array\n");
	return -1;
    }
    printf("init_array:\n");
    print_uint64(&mtgp64, 1000);
    mtgp64_free(&mtgp64);

    rc = mtgp64_init_by_str(&mtgp64, params, seed_str);
    if (rc) {
	printf("failure in mtgp64_init_by_str\n");
	return -1;
    }
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
