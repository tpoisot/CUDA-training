#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <inttypes.h>
#include "mtgp32-fast.h"
#include "mtgp64-fast.h"

#define SIZE 1000
static int CNT[256];

static void make_table(void);
inline static int count32bit(uint32_t r);
inline static int count64bit(uint64_t r);
static void set_state32(uint32_t array[], int size, int mexp, int pos);
static void set_state64(uint64_t array[], int size, int mexp, int pos);
static int u32_zeroex_count(int mexp, int no, int count);
static int u64_zeroex_count(int mexp, int no, int count);

int main(int argc, char *argv[]) {
    int bit_size;
    int mexp;
    int no;
    int count;

    if (argc <= 4) {
	printf("%s: bit_size mexp no. count", argv[0]);
	return 1;
    }
    bit_size = strtol(argv[1], NULL, 10);
    if (errno) {
	printf("%s: bit_size error.", argv[0]);
	return 1;
    }
    if (bit_size != 32 && bit_size != 64) {
	printf("%s: bit_size error. bit size is 32 or 64", argv[0]);
	return 1;
    }
    mexp = strtol(argv[2], NULL, 10);
    if (errno) {
	printf("%s: mexp error.", argv[0]);
	return 1;
    }
    no = strtol(argv[3], NULL, 10);
    if (errno) {
	printf("%s: no. error.", argv[0]);
	return 3;
    }
    count = (int)strtol(argv[4], NULL, 10);
    if (errno) {
	printf("%s: count error:%s\n", argv[0], argv[4]);
	return -1;
    }
    if (bit_size == 32) {
	u32_zeroex_count(mexp, no, count);
    } else {
	u64_zeroex_count(mexp, no, count);
    }
    return 0;
}

static void make_table(void) {
    int i, j;
    int bit;
    int mask;

    for (i = 0; i < 256; i++) {
	mask = 1;
	bit = 0;
	for (j = 0; j < 8; j++) {
	    if (i & mask) {
		bit++;
	    }
	    mask = mask << 1;
	}
	CNT[i] = bit;
    }
}

inline static int count32bit(uint32_t r) {
    int i;
    int sum;

    sum = 0;
    for (i = 0; i < 4; i++) {
	sum += CNT[r & 0xff];
	r = r >> 8;
    }
    return sum;
}

inline static int count64bit(uint64_t r) {
    int i;
    int sum;

    sum = 0;
    for (i = 0; i < 8; i++) {
	sum += CNT[r & 0xff];
	r = r >> 8;
    }
    return sum;
}

static void set_state32(uint32_t array[], int size, int mexp, int pos) {
    const int bit_size = 32;
    int remain = mexp % bit_size;

    memset(array, 0, sizeof(uint32_t) * size);
    if (pos < remain) {
	array[0] = 1 << (bit_size - remain + pos);
    } else {
	pos = pos - remain;
	array[pos / bit_size + 1] = 1 << (pos % bit_size);
    }
}

static void set_state64(uint64_t array[], int size, int mexp, int pos) {
    const int bit_size = 64;
    int remain = mexp % bit_size;

    memset(array, 0, sizeof(uint64_t) * size);
    if (pos < remain) {
	array[0] = 1 << (bit_size - remain + pos);
    } else {
	pos = pos - remain;
	array[pos / bit_size + 1] = 1 << (pos % bit_size);
    }
}

static int u32_zeroex_count(int mexp, int no, int count) {
    int i, j;
    uint32_t ran;
    int ave[count + 1];
    int st1000[SIZE];
    int bits;
    int sum;
    int rc;
    int seed = 1;
    mtgp32_fast_t mtgp32;
    mtgp32_params_fast_t *params;

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
	printf("mexp shuould be 11213, 23209 or 44497\n");
	exit(1);
    }
    if (no >= 128 || no < 0) {
	printf("No. must be between 0 and 127\n");
	exit(1);
    }
    params += no;
    rc = mtgp32_init(&mtgp32, params, seed);
    if (rc) {
	printf("failure in mtgp32_init\n");
	exit(1);
    }
    mtgp32_print_idstring(&mtgp32, stdout);

    memset(ave, 0, sizeof(ave));
    make_table();

    for (i = 0; i < mexp; i++) {
	set_state32(mtgp32.status->array, mtgp32.status->size, mexp, i);
	mtgp32.status->idx = mtgp32.status->size -1;
	sum = 0;
	for (j = 0; j < SIZE; j++) {
	    ran = mtgp32_genrand_uint32(&mtgp32);
	    ran = mtgp32.status->array[mtgp32.status->idx];
	    bits = count32bit(ran);
	    st1000[j] = bits;
	    sum += bits;
	}
	ave[0] += sum;
	for (j = 0; j < count; j++) {
	    ran = mtgp32_genrand_uint32(&mtgp32);
	    ran = mtgp32.status->array[mtgp32.status->idx];
	    bits = count32bit(ran);
	    sum = sum - st1000[j % SIZE] + bits;
	    st1000[j % SIZE] = bits;
	    ave[j + 1] += sum;
	}
    }
    for (i = 0; i <= count; i++) {
	printf("%d, %.4f\n", i, (double)ave[i] / (32 * SIZE) / mexp);
    }
    return 0;
}

static int u64_zeroex_count(int mexp, int no, int count) {
    int i, j;
    uint64_t ran;
    int ave[count + 1];
    int st1000[SIZE];
    int bits;
    int sum;
    int rc;
    int seed = 1;
    mtgp64_fast_t mtgp64;
    mtgp64_params_fast_t *params;

    switch (mexp) {
    case 23209:
	params = mtgp64_params_fast_23209;
	break;
    case 44497:
	params = mtgp64_params_fast_44497;
	break;
    case 110503:
	params = mtgp64_params_fast_110503;
	break;
    default:
	printf("mexp shuould be 23209, 44497 or 110503\n");
	exit(1);
    }
    if (no >= 128 || no < 0) {
	printf("No. must be between 0 and 127\n");
	exit(1);
    }
    params += no;
    rc = mtgp64_init(&mtgp64, params, seed);
    if (rc) {
	printf("failure in mtgp64_init\n");
	exit(1);
    }
    mtgp64_print_idstring(&mtgp64, stdout);

    memset(ave, 0, sizeof(ave));
    make_table();

    for (i = 0; i < mexp; i++) {
	set_state64(mtgp64.status->array, mtgp64.status->size, mexp, i);
	mtgp64.status->idx = mtgp64.status->size -1;
	sum = 0;
	for (j = 0; j < SIZE; j++) {
	    ran = mtgp64_genrand_uint64(&mtgp64);
	    bits = count64bit(ran);
	    st1000[j] = bits;
	    sum += bits;
	}
	ave[0] += sum;
	for (j = 0; j < count; j++) {
	    ran = mtgp64_genrand_uint64(&mtgp64);
	    bits = count64bit(ran);
	    sum = sum - st1000[j % SIZE] + bits;
	    st1000[j % SIZE] = bits;
	    ave[j + 1] += sum;
	}
    }
    for (i = 0; i <= count; i++) {
	printf("%d, %.4f\n", i, (double)ave[i] / (64 * SIZE) / mexp);
    }
    return 0;
}
