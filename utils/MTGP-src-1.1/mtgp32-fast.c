/**
 * @file mtgp32-fast.c
 *
 * @brief Mersenne Twister for Graphic Processors (mtgp32), which
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
#include "mtgp32-fast.h"

static uint32_t ini_func1(uint32_t x);
static uint32_t ini_func2(uint32_t x);
static int alloc_state(mtgp32_fast_t *mtgp32, const mtgp32_params_fast_t *para);
static const uint32_t non_zero = 0x4d544750;

/**
 * \private
 * This function allocates the internal state vector.
 *
 * @param[in,out] mtgp32 MTGP all in one structure.
 * @param[in] para a parameter structure.
 * @return allocation status. if 0 O.K.
 */
static int alloc_state(mtgp32_fast_t *mtgp32,
		       const mtgp32_params_fast_t *para) {
    int size = para->mexp / 32 + 1;
    int large_size;
    mtgp32_status_fast_t *st;

    int x;
    int y = size;
    for (x = 1; (x != size) && (y > 0); x <<= 1, y >>= 1);
    large_size = x;

    st = (mtgp32_status_fast_t *)malloc(sizeof(mtgp32_status_fast_t)
				     + sizeof(uint32_t) * large_size);
    if (st == NULL) {
	return -1;
    }
    mtgp32->params = *para;
    mtgp32->status = st;
    st->size = size;
    st->large_size = large_size;
    st->large_mask = large_size -1;
    return 0;
}

/**
 * This function represents a function used in the initialization
 * by mtgp32_init_by_array() and mtgp32_init_by_str().
 * @param[in] x 32-bit integer
 * @return 32-bit integer
 */
static uint32_t ini_func1(uint32_t x) {
    return (x ^ (x >> 27)) * UINT32_C(1664525);
}

/**
 * This function represents a function used in the initialization
 * by mtgp32_init_by_array() and mtgp32_init_by_str().
 * @param[in] x 32-bit integer
 * @return 32-bit integer
 */
static uint32_t ini_func2(uint32_t x) {
    return (x ^ (x >> 27)) * UINT32_C(1566083941);
}
/*----------------
  PUBLIC FUNCTIONS
  ----------------*/
/**
 * \public
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
int mtgp32_init(mtgp32_fast_t *mtgp32,
		const mtgp32_params_fast_t *para, uint32_t seed) {
    int rc;
    rc = alloc_state(mtgp32, para);
    if (rc) {
	return rc;
    }
    mtgp32->status->idx = mtgp32->status->size - 1;
    mtgp32_init_state(mtgp32->status->array, para, seed);
    return 0;
}

/**
 * This function initializes the internal state array with a 32-bit
 * integer seed. The allocated memory should be freed by calling
 * mtgp32_free(). \b para should be one of the elements in the
 * parameter table (mtgp32-param-ref.c).
 *
 * This function is call by cuda program, because cuda program uses
 * another structure and another allocation method.
 *
 * @param[out] array MTGP internal status vector.
 * @param[in] para parameter structure
 * @param[in] seed a 32-bit integer used as the seed.
 */
void mtgp32_init_state(uint32_t array[],
		      const mtgp32_params_fast_t *para, uint32_t seed) {
    int i;
    int size = para->mexp / 32 + 1;
    uint32_t hidden_seed;
    uint32_t tmp;
    hidden_seed = para->tbl[4] ^ (para->tbl[8] << 16);
    tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(array, tmp & 0xff, sizeof(uint32_t) * size);
    array[0] = seed;
    array[1] = hidden_seed;
    for (i = 1; i < size; i++) {
	array[i] ^= UINT32_C(1812433253) * (array[i - 1]
					    ^ (array[i - 1] >> 30))
	    + i;
    }
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
int mtgp32_init_by_array(mtgp32_fast_t *mtgp32,
			 const mtgp32_params_fast_t *para,
			 uint32_t *array, int length) {
    int i, j, count;
    uint32_t r;
    int lag;
    int mid;
    int size = para->mexp / 32 + 1;
    uint32_t hidden_seed;
    uint32_t tmp;
    mtgp32_status_fast_t *st;
    int rc;

    rc = alloc_state(mtgp32, para);
    if (rc) {
	return rc;
    }

    st = mtgp32->status;
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
 * @return memory allocation result. if 0 then O.K.
 */
int mtgp32_init_by_str(mtgp32_fast_t *mtgp32,
		       const mtgp32_params_fast_t *para, char *array) {
    int i, j, count;
    uint32_t r;
    int lag;
    int mid;
    int size = para->mexp / 32 + 1;
    int length = strlen(array);
    uint32_t hidden_seed;
    uint32_t tmp;
    mtgp32_status_fast_t *st;
    int rc;

    rc = alloc_state(mtgp32, para);
    if (rc) {
	return rc;
    }

    st = mtgp32->status;
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
void mtgp32_free(mtgp32_fast_t *mtgp32) {
    free(mtgp32->status);
}

/**
 * This function prints the Mersenne exponent and SHA1 of characteristic
 * polynomial of generators state transition function.
 *
 * @param[in] mtgp32 MTGP all in one structure.
 * @param[in,out] fp FILE pointer.
 */
void mtgp32_print_idstring(const mtgp32_fast_t *mtgp32, FILE *fp) {
    int i;
    fprintf(fp, "mtgp32:%d:", mtgp32->params.mexp);
    for (i = 0; i < 20; i++) {
	fprintf(fp, "%02x", (unsigned int)mtgp32->params.poly_sha1[i]);
    }
    fprintf(fp, "\n");
}

#if defined(MAIN)
#include <errno.h>
void print_uint32(mtgp32_fast_t *mtgp32, int count);
void print_close1_open2(mtgp32_fast_t *mtgp32, int count);
void print_close_open(mtgp32_fast_t *mtgp32, int count);
void print_open_close(mtgp32_fast_t *mtgp32, int count);
void print_open_open(mtgp32_fast_t *mtgp32, int count);

void print_uint32(mtgp32_fast_t *mtgp32, int count) {
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

void print_close1_open2(mtgp32_fast_t *mtgp32, int count) {
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

void print_close_open(mtgp32_fast_t *mtgp32, int count) {
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

void print_open_close(mtgp32_fast_t *mtgp32, int count) {
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

void print_open_open(mtgp32_fast_t *mtgp32, int count) {
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
    mtgp32_params_fast_t *params;
    mtgp32_fast_t mtgp32;
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
    case 11213:
	params = mtgp32_params_fast_11213;
	break;
    case 23209:
	params = mtgp32_params_fast_23209;
	break;
    case 44497:
	params = mtgp32_params_fast_44497;
	break;
    default:
	printf("%s: mexp no.\n", argv[0]);
	printf("mexp shuould be 11213, 23209 or 44497\n");
	return 4;
    }
    if (no >= 128 || no < 0) {
	printf("%s: mexp no.\n", argv[0]);
	printf("no must be between 0 and 127\n");
	return 5;
    }
    params += no;
    rc = mtgp32_init(&mtgp32, params, seed);
    if (rc) {
	printf("failure in mtgp32_init\n");
	return -1;
    }
    mtgp32_print_idstring(&mtgp32, stdout);
    printf("init:\n");
    print_uint32(&mtgp32, 1000);
    mtgp32_free(&mtgp32);

    rc = mtgp32_init_by_array(&mtgp32, params, seed_ar, 4);
    if (rc) {
	printf("failure in mtgp32_init_by_array\n");
	return -1;
    }
    printf("init_array:\n");
    print_uint32(&mtgp32, 1000);
    mtgp32_free(&mtgp32);

    rc = mtgp32_init_by_str(&mtgp32, params, seed_str);
    if (rc) {
	printf("failure in mtgp32_init_by_str\n");
	return -1;
    }
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
