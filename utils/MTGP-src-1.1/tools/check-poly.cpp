/**
 * This program uses NTL:
 * http://shoup.net/ntl/
 */
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <NTL/GF2X.h>
#include <NTL/vec_GF2.h>
#include <NTL/GF2XFactoring.h>
#include <errno.h>
#include <openssl/sha.h>
extern "C" {
#include "mtgp32-fast.h"
#include "mtgp64-fast.h"
}

using namespace NTL;
using namespace std;

static void u32_poly_check(int mexp, int no, int seed);
static void u64_poly_check(int mexp, int no, int seed);
static void poly_sha1(string& str, unsigned char sha1[], const GF2X& poly);
static void poly_check(vec_GF2& vec, int mexp, unsigned char param_sha1[]);

int main(int argc, char *argv[]) {
    int bit_size;
    int mexp;
    int no;
    uint32_t seed = 1;

    if (argc <= 3) {
	cout << argv[0] << ": bit_size mexp no." << endl;
	return 1;
    }
    bit_size = strtol(argv[1], NULL, 10);
    if (errno) {
	cout << argv[0] << ": bit_size error." << endl;
	return 1;
    }
    if (bit_size != 32 && bit_size != 64) {
	cout << argv[0] << ": bit_size error. bit size is 32 or 64" << endl;
	return 1;
    }
    mexp = strtol(argv[2], NULL, 10);
    if (errno) {
	cout << argv[0] << ": mexp no." << endl;
	return 1;
    }
    no = strtol(argv[3], NULL, 10);
    if (errno) {
	cout << argv[0] << ": mexp no.\n" << endl;
	return 3;
    }
    if (bit_size == 32) {
	u32_poly_check(mexp, no, seed);
    } else {
	u64_poly_check(mexp, no, seed);
    }
    return 0;
}

static void poly_sha1(string& str, unsigned char sha1[], const GF2X& poly) {
    SHA_CTX ctx;
    SHA1_Init(&ctx);
    if (deg(poly) < 0) {
	SHA1_Update(&ctx, "-1", 2);
    }
    for(int i = 0; i <= deg(poly); i++) {
	if(rep(coeff(poly, i)) == 1) {
	    SHA1_Update(&ctx, "1", 1);
	} else {
	    SHA1_Update(&ctx, "0", 1);
	}
    }
    unsigned char md[SHA_DIGEST_LENGTH];
    SHA1_Final(md, &ctx);
    stringstream ss;
    for (int i = 0; i < SHA_DIGEST_LENGTH; i++) {
	ss << setfill('0') << setw(2) << hex
	   << static_cast<int>(md[i]);
	sha1[i] = md[i];
    }
    sha1[SHA_DIGEST_LENGTH] = 0;
    ss >> str;
}

static void poly_check(vec_GF2& vec, int mexp, unsigned char param_sha1[]) {
    GF2X poly;
    string s;
    unsigned char sha1[SHA_DIGEST_LENGTH + 1];

    MinPolySeq(poly, vec, mexp);
    poly_sha1(s, sha1, poly);
    cout << s << endl;
    if (strcmp((char *)sha1, (char *)param_sha1) == 0) {
	cout << "poly sha1 OK" << endl;
    } else {
	cout << "poly sha1 NG" << endl;
    }
    if (deg(poly) == mexp) {
	cout << "poly deg = " << deg(poly) << " OK" << endl;
    } else {
	cout << "poly deg = " << deg(poly) << " NG" << endl;
    }
    if (IterIrredTest(poly)) {
	cout << "poly irreducible OK" << endl;
    } else {
	cout << "poly irreducible OK" << endl;
    }
}

static void u32_poly_check(int mexp, int no, int seed) {
    mtgp32_params_fast_t *params;
    mtgp32_fast_t mtgp32;
    vec_GF2 vec;
    int rc;

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
	cout << "mexp shuould be 11213, 23209 or 44497" << endl;
	exit(1);
    }
    if (no >= 128 || no < 0) {
	cout << "no must be between 0 and 127" << endl;
	exit(1);
    }
    params += no;
    rc = mtgp32_init(&mtgp32, params, seed);
    if (rc) {
	cout << "failure in mtgp32_init" << endl;
	exit(1);
    }
    mtgp32_print_idstring(&mtgp32, stdout);
    vec.SetLength(2 * mexp);
    for (int i = 0; i < 2 * mexp; i++) {
	vec[i] = mtgp32_genrand_uint32(&mtgp32) & 1;
    }
    poly_check(vec, mexp, mtgp32.params.poly_sha1);
    mtgp32_free(&mtgp32);
}

static void u64_poly_check(int mexp, int no, int seed) {
    mtgp64_params_fast_t *params;
    mtgp64_fast_t mtgp64;
    vec_GF2 vec;
    int rc;

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
	cout << "mexp shuould be 11213, 23209 or 44497" << endl;
	exit(1);
    }
    if (no >= 128 || no < 0) {
	cout << "no must be between 0 and 127" << endl;
	exit(1);
    }
    params += no;
    rc = mtgp64_init(&mtgp64, params, seed);
    if (rc) {
	cout << "failure in mtgp64_init." << endl;
	exit(1);
    }
    mtgp64_print_idstring(&mtgp64, stdout);
    vec.SetLength(2 * mexp);
    for (int i = 0; i < 2 * mexp; i++) {
	vec[i] = mtgp64_genrand_uint64(&mtgp64) & 1;
    }
    poly_check(vec, mexp, mtgp64.params.poly_sha1);
    mtgp64_free(&mtgp64);
}
