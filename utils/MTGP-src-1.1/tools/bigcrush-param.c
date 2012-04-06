/**
 * This is a test progam using TestU01:
 * http://www.iro.umontreal.ca/~simardr/testu01/tu01.html
 */
#include <stdlib.h>
#include <errno.h>
#include "unif01.h"
#include "bbattery.h"
#include "mtgp32-fast.h"
#include "mtgp64-fast.h"
#include <inttypes.h>
#include <stdint.h>

static mtgp32_fast_t mtgp32;
static mtgp64_fast_t mtgp64;
#if 0
static int rep[200] = {0};
#endif
uint32_t mtgp32_uint32(void);
uint32_t mtgp64_uint32(void);
void u32_big_crush(mtgp32_params_fast_t *params, int seed);
void u64_big_crush(mtgp64_params_fast_t *params, int seed);
void read_params32(mtgp32_params_fast_t *params, char * argv[]);
void read_params64(mtgp64_params_fast_t *params, char * argv[]);
void fill_table32(uint32_t dest[], uint32_t source[], int size);
void fill_table64(uint64_t dest[], uint64_t source[], int size);

int main(int argc, char *argv[]) {
    int bit_size;
    int seed = 1;

#if 0
    rep[34] = 0;
    rep[35] = 1;
    rep[99] = 0;
    rep[100] = 1;
#endif

    if (argc <= 15) {
	printf("%s: bit_size mexp num [seed]\n", argv[0]);
	return 1;
    }
    bit_size = strtol(argv[1], NULL, 10);
    if (errno) {
	printf("%s: bit_size error.\n", argv[0]);
	return 1;
    }
    if (bit_size != 32 && bit_size != 64) {
	printf("%s: bit_size error. bit size is 32 or 64\n", argv[0]);
	return 1;
    }
    if (argc >= 17) {
	seed = strtol(argv[16], NULL, 10);
	if (errno) {
	    printf("%s: seed error.\n", argv[0]);
	    return 1;
	}
    }
    if (bit_size == 32) {
	mtgp32_params_fast_t p32;
	read_params32(&p32, &argv[2]);
	u32_big_crush(&p32, seed);
    } else {
	mtgp64_params_fast_t p64;
	read_params64(&p64, &argv[2]);
	u64_big_crush(&p64, seed);
    }
    return 0;
}


uint32_t mtgp32_uint32(void) {
    return mtgp32_genrand_uint32(&mtgp32);
}

uint32_t mtgp64_uint32(void) {
    return (uint32_t)(mtgp64_genrand_uint64(&mtgp64) & 0xffffffffU);
}

void read_params32(mtgp32_params_fast_t *params, char *argv[]) {
    uint32_t tmp[4];
    uint32_t id;
    params->mexp = strtol(argv[0], NULL, 10);
    params->pos = strtol(argv[1], NULL, 10);
    params->sh1 = strtol(argv[2], NULL, 10);
    params->sh2 = strtol(argv[3], NULL, 10);
    id = strtol(argv[4], NULL, 10);
    params->mask = strtoul(argv[5], NULL, 16);
    for (int i = 0; i < 4; i++) {
	tmp[i] = strtoul(argv[i + 6], NULL, 16);
    }
    fill_table32(params->tbl, tmp, 16);
    for (int i = 0; i < 4; i++) {
	tmp[i] = strtoul(argv[i + 10], NULL, 16);
    }
    fill_table32(params->tmp_tbl, tmp, 16);
    for (int i = 0; i < 16; i++) {
	params->flt_tmp_tbl[i] = 0x3f800000U | (params->tmp_tbl[i] >> 9);
    }
    snprintf((char *)params->poly_sha1, sizeof(params->poly_sha1), "%d", id);
#if 1
    printf("mexp:%d\n", params->mexp);
    printf("pos:%d\n", params->pos);
    printf("sh1:%d\n", params->sh1);
    printf("sh2:%d\n", params->sh2);
    printf("mask:%08x\n", params->mask);
    for (int i = 0; i < 16; i++) {
	printf("tbl[%02d]:%08x\n", i, params->tbl[i]);
    }
    for (int i = 0; i < 16; i++) {
	printf("tmp_tbl[%02d]:%08x\n", i, params->tmp_tbl[i]);
    }
    for (int i = 0; i < 16; i++) {
	printf("flt_tmp_tbl[%02d]:%08x\n", i, params->flt_tmp_tbl[i]);
    }
    fflush(stdout);
#endif
}


void read_params64(mtgp64_params_fast_t *params, char *argv[]) {
    uint64_t tmp[4];
    uint32_t id;
    params->mexp = strtol(argv[0], NULL, 10);
    params->pos = strtol(argv[1], NULL, 10);
    params->sh1 = strtol(argv[2], NULL, 10);
    params->sh2 = strtol(argv[3], NULL, 10);
    id = strtol(argv[4], NULL, 10);
    params->mask = strtoull(argv[5], NULL, 16);
    for (int i = 0; i < 4; i++) {
	tmp[i] = strtoull(argv[i + 6], NULL, 16);
    }
    fill_table64(params->tbl, tmp, 16);
    for (int i = 0; i < 4; i++) {
	tmp[i] = strtoull(argv[i + 10], NULL, 16);
    }
    fill_table64(params->tmp_tbl, tmp, 16);
    for (int i = 0; i < 16; i++) {
	params->dbl_tmp_tbl[i] = UINT64_C(0x3ff0000000000000)
	    | (params->tmp_tbl[i] >> 12);
    }
    memset(params->poly_sha1, 0, sizeof(params->poly_sha1));
    snprintf((char *)params->poly_sha1, sizeof(params->poly_sha1), "%d", id);
#if 1
    printf("mexp:%d\n", params->mexp);
    printf("pos:%d\n", params->pos);
    printf("sh1:%d\n", params->sh1);
    printf("sh2:%d\n", params->sh2);
    printf("mask:%016"PRIx64"\n", params->mask);
    for (int i = 0; i < 16; i++) {
	printf("tbl[%02d]:%016"PRIx64"\n", i, params->tbl[i]);
    }
    for (int i = 0; i < 16; i++) {
	printf("tmp_tbl[%02d]:%016"PRIx64"\n", i, params->tmp_tbl[i]);
    }
    for (int i = 0; i < 16; i++) {
	printf("dbl_tmp_tbl[%02d]:%016"PRIx64"\n", i, params->dbl_tmp_tbl[i]);
    }
    fflush(stdout);
#endif
}

void fill_table32(uint32_t dest[], uint32_t source[], int size) {
    for(int i = 0; i < size; i++) {
	dest[i] = 0;
	for(int j = 1, k = 0; j <= i; j <<= 1, k++) {
	    if (i & j) {
		dest[i] ^= source[k];
	    }
	}
    }
}

void fill_table64(uint64_t dest[], uint64_t source[], int size) {
    for(int i = 0; i < size; i++) {
	dest[i] = 0;
	for(int j = 1, k = 0; j <= i; j <<= 1, k++) {
	    if (i & j) {
		dest[i] ^= source[k];
	    }
	}
    }
}

void u32_big_crush(mtgp32_params_fast_t *params, int seed) {
    unif01_Gen *gen;
    int rc;
    rc = mtgp32_init(&mtgp32, params, seed);
    if (rc) {
	printf("failure in mtgp32_init\n");
	exit(1);
    }
    mtgp32_print_idstring(&mtgp32, stdout);
    gen = unif01_CreateExternGenBits ("MTGP", mtgp32_uint32);
    /* bbattery_RepeatBigCrush(gen, rep); */
    bbattery_BigCrush(gen);
    unif01_DeleteExternGenBits(gen);
    mtgp32_free(&mtgp32);
}

void u64_big_crush(mtgp64_params_fast_t *params, int seed) {
    unif01_Gen *gen;
    int rc;
    rc = mtgp64_init(&mtgp64, params, seed);
    if (rc) {
	printf("failure in mtgp64_init\n");
	exit(1);
    }
    mtgp64_print_idstring(&mtgp64, stdout);

    printf("MTGP64 seed = %d\n", seed);
    fflush(stdout);
    /* gen = unif01_CreateExternGen01("MTGP64", mtgp64_double); */
    gen = unif01_CreateExternGenBits("MTGP64", mtgp64_uint32);
    /* bbattery_RepeatBigCrush(gen, rep); */
    bbattery_BigCrush(gen);
    unif01_DeleteExternGen01(gen);

    mtgp64_free(&mtgp64);
}
