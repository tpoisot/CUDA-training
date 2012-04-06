#ifndef MTGP32_REF_H
#define MTGP32_REF_H
/**
 * @file mtgp32-ref.h
 * @brief Mersenne Twister for Graphic Processors (MTGP), which
 * generates 32-bit unsigned integers and float precision floating
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
 * MTGP32 parameters.
 */
typedef struct MTGP32_PARAMS_REF_T {
    int mexp;
    int pos;
    int sh1;
    int sh2;
    uint32_t tbl[4];
    uint32_t tmp_tbl[4];
    uint32_t mask;
    unsigned char poly_sha1[21];
} mtgp32_params_ref_t;

/**
 * MTGP32 internal state array
 */
typedef struct MTGP32_STATUS_REF_T {
    int size;
    int idx;
    uint32_t array[0];
} mtgp32_status_ref_t;

/**
 * MTGP32 structure
 */
typedef struct MTGP32_REF_T {
    mtgp32_params_ref_t params;
    mtgp32_status_ref_t *status;
} mtgp32_ref_t;

/** parameter constants tables for MEXP=11213 */
extern mtgp32_params_ref_t mtgp32_params_ref_11213[128];
/** parameter constants tables for MEXP=23209 */
extern mtgp32_params_ref_t mtgp32_params_ref_23209[128];
/** parameter constants tables for MEXP=44497 */
extern mtgp32_params_ref_t mtgp32_params_ref_44497[128];

int mtgp32_init(mtgp32_ref_t *mtgp32,
		const mtgp32_params_ref_t *para, uint32_t seed);
int mtgp32_init_by_array(mtgp32_ref_t *mtgp32,
			 const mtgp32_params_ref_t *para,
			 const uint32_t *array, int length);
int mtgp32_init_by_str(mtgp32_ref_t *mtgp32,
		       const mtgp32_params_ref_t *para,
		       const char *str);
void mtgp32_free(mtgp32_ref_t *mtgp32);
uint32_t mtgp32_genrand_uint32(mtgp32_ref_t *mtgp32);
float mtgp32_genrand_close1_open2(mtgp32_ref_t *mtgp32);
float mtgp32_genrand_close_open(mtgp32_ref_t *mtgp32);
float mtgp32_genrand_open_close(mtgp32_ref_t *mtgp32);
float mtgp32_genrand_open_open(mtgp32_ref_t *mtgp32);
void mtgp32_print_idstring(const mtgp32_ref_t *mtgp32, FILE *fp);

#endif
