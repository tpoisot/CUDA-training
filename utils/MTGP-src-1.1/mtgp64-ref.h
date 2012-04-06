#ifndef MTGP64_REF_H
#define MTGP64_REF_H
/**
 * @file mtgp64-ref.h
 * @brief Mersenne Twister for Graphic Processors (MTGP), which
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
#include <stdint.h>

/**
 * MTGP64 parameters.
 */
typedef struct MTGP64_PARAMS_REF_T {
    int mexp;
    int pos;
    int sh1;
    int sh2;
    uint64_t tbl[4];
    uint64_t tmp_tbl[4];
    uint64_t mask;
    unsigned char poly_sha1[21];
} mtgp64_params_ref_t;

/**
 * MTGP64 internal state array
 */
typedef struct MTGP64_STATUS_REF_T {
    int size;
    int idx;
    uint64_t array[0];
} mtgp64_status_ref_t;

/**
 * MTGP64 structure
 */
typedef struct MTGP64_REF_T {
    mtgp64_params_ref_t params;
    mtgp64_status_ref_t *status;
} mtgp64_ref_t;

/** parameter constants tables for MEXP=23209 */
extern mtgp64_params_ref_t mtgp64_params_ref_23209[128];
/** parameter constants tables for MEXP=44497 */
extern mtgp64_params_ref_t mtgp64_params_ref_44497[128];
/** parameter constants tables for MEXP=110503 */
extern mtgp64_params_ref_t mtgp64_params_ref_110503[128];

int mtgp64_init(mtgp64_ref_t *mtgp64,
		const mtgp64_params_ref_t *para, uint64_t seed);
int mtgp64_init_by_array(mtgp64_ref_t *mtgp64,
			 const mtgp64_params_ref_t *para,
			 const uint64_t *array, int length);
int mtgp64_init_by_str(mtgp64_ref_t *mtgp64,
		       const mtgp64_params_ref_t *para,
		       const char *str);
void mtgp64_free(mtgp64_ref_t *mtgp64);
uint64_t mtgp64_genrand_uint64(mtgp64_ref_t *mtgp64);
double mtgp64_genrand_close1_open2(mtgp64_ref_t *mtgp64);
double mtgp64_genrand_close_open(mtgp64_ref_t *mtgp64);
double mtgp64_genrand_open_close(mtgp64_ref_t *mtgp64);
double mtgp64_genrand_open_open(mtgp64_ref_t *mtgp64);
void mtgp64_print_idstring(const mtgp64_ref_t *mtgp64, FILE *fp);

#endif
