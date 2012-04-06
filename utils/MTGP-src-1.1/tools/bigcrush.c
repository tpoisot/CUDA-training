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

static mtgp32_fast_t mtgp32;
static mtgp64_fast_t mtgp64;

uint32_t mtgp32_uint32(void);
double mtgp64_double(void);
void u32_big_crush(int mexp, int num, int seed);
void u64_big_crush(int mexp, int num, int seed);

int main(int argc, char *argv[]) {
    int bit_size;
    int mexp;
    int num;
    int seed = 1;

    if (argc <= 3) {
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
    mexp = strtol(argv[2], NULL, 10);
    if (errno) {
	printf("%s: mexp error.\n", argv[0]);
	return 1;
    }
    num = strtol(argv[3], NULL, 10);
    if (errno) {
	printf("%s: num error.\n", argv[0]);
	return 1;
    }
    if (argc >= 5) {
	seed = strtol(argv[4], NULL, 10);
	if (errno) {
	    printf("%s: seed error.\n", argv[0]);
	    return 1;
	}
    }
    if (bit_size == 32) {
	u32_big_crush(mexp, num, seed);
    } else {
	u64_big_crush(mexp, num, seed);
    }
    return 0;
}


uint32_t mtgp32_uint32(void) {
    return mtgp32_genrand_uint32(&mtgp32);
}

double mtgp64_double(void) {
    return mtgp64_genrand_close_open(&mtgp64);
}

void u32_big_crush(int mexp, int num, int seed) {
    unif01_Gen *gen;
    int rc;
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
    if (num >= 128 || num < 0) {
	printf("num must be between 0 and 127\n");
	exit(1);
    }
    params += num;
    rc = mtgp32_init(&mtgp32, params, seed);
    if (rc) {
	printf("failure in mtgp32_init\n");
	exit(1);
    }
    mtgp32_print_idstring(&mtgp32, stdout);
    /*
      gen = unif01_CreateExternGen01("dSFMT", dSFMT);
      bbattery_BigCrush(gen);
      unif01_DeleteExternGen01(gen);
    */
    gen = unif01_CreateExternGenBits ("MTGP", mtgp32_uint32);
    bbattery_BigCrush (gen);
    unif01_DeleteExternGenBits (gen);
    mtgp32_free(&mtgp32);
}

void u64_big_crush(int mexp, int num, int seed) {
    unif01_Gen *gen;
    int rc;
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
    if (num >= 128 || num < 0) {
	printf("num must be between 0 and 127\n");
	exit(1);
    }
    params += num;
    rc = mtgp64_init(&mtgp64, params, seed);
    if (rc) {
	printf("failure in mtgp64_init\n");
	exit(1);
    }
    mtgp64_print_idstring(&mtgp64, stdout);

    gen = unif01_CreateExternGen01("MTGP", mtgp64_double);
    bbattery_BigCrush(gen);
    unif01_DeleteExternGen01(gen);

    mtgp64_free(&mtgp64);
}
