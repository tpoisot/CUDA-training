/*
 * Sample Program for CUDA 2.3
 * written by M.Saito (saito@math.sci.hiroshima-u.ac.jp)
 *
 * This sample uses texture reference.
 * The generation speed of PRNG using texture is faster than using
 * constant tabel on Geforce GTX 260.
 *
 * MTGP32-11213
 * This program generates 32-bit unsigned integers.
 * The period of generated integers is 2<sup>11213</sup>-1.
 * This also generates single precision floating point numbers.
 */

#include <stdio.h>
#include <cuda.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <stdlib.h>

#include "mtgp-util.cuh"
#include "mtgp32-fast.h"

#define MTGPDC_MEXP 11213
#define MTGPDC_N 351
#define MTGPDC_FLOOR_2P 256
#define MTGPDC_CEIL_2P 512
#define MTGPDC_PARAM_TABLE mtgp32dc_params_fast_11213
#define MEXP 11213
#define THREAD_NUM MTGPDC_FLOOR_2P
#define LARGE_SIZE (THREAD_NUM * 3)
#define BLOCK_NUM_MAX 200
#define TBL_SIZE 16
#define N MTGPDC_N

extern mtgp32_params_fast_t mtgp32dc_params_fast_11213[];

/**
 * kernel I/O
 * This structure must be initialized before first use.
 */
struct mtgp32_kernel_status_t {
    uint32_t status[MTGPDC_N];
};

texture<uint32_t, 1, cudaReadModeElementType> tex_param_ref;
texture<uint32_t, 1, cudaReadModeElementType> tex_temper_ref;
texture<uint32_t, 1, cudaReadModeElementType> tex_single_ref;

/*
 * Generator Parameters.
 */
__constant__ uint32_t pos_tbl[BLOCK_NUM_MAX];
__constant__ uint32_t sh1_tbl[BLOCK_NUM_MAX];
__constant__ uint32_t sh2_tbl[BLOCK_NUM_MAX];
__constant__ uint32_t mask[1];

/**
 * Shared memory
 * The generator's internal status vector.
 */
__shared__ uint32_t status[LARGE_SIZE];

/**
 * The function of the recursion formula calculation.
 *
 * @param[in] X1 the farthest part of state array.
 * @param[in] X2 the second farthest part of state array.
 * @param[in] Y a part of state array.
 * @param[in] bid block id.
 * @return output
 */
__device__ uint32_t para_rec(uint32_t X1, uint32_t X2, uint32_t Y, int bid) {
    uint32_t X = (X1 & mask[0]) ^ X2;
    uint32_t MAT;

    X ^= X << sh1_tbl[bid];
    Y = X ^ (Y >> sh2_tbl[bid]);
    MAT = tex1Dfetch(tex_param_ref, bid * 16 + (Y & 0x0f));
    return Y ^ MAT;
}

/**
 * The tempering function.
 *
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @param[in] bid block id.
 * @return the tempered value.
 */
__device__ uint32_t temper(uint32_t V, uint32_t T, int bid) {
    uint32_t MAT;

    T ^= T >> 16;
    T ^= T >> 8;
    MAT = tex1Dfetch(tex_temper_ref, bid * 16 + (T & 0x0f));
    return V ^ MAT;
}

/**
 * The tempering and converting function.
 * By using the preset-ted table, converting to IEEE format
 * and tempering are done simultaneously.
 * Resulted outputs are distributed in the range [1, 2).
 *
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @param[in] bid block id.
 * @return the tempered and converted value.
 */
__device__ float temper_single(uint32_t V, uint32_t T, int bid) {
    uint32_t MAT;
    uint32_t r;

    T ^= T >> 16;
    T ^= T >> 8;
    MAT = tex1Dfetch(tex_single_ref, bid * 16 + (T & 0x0f));
    r = (V >> 9) ^ MAT;
    return __int_as_float(r);
}

/**
 * The tempering and converting function.
 * By using the preset-ted table, converting to IEEE format
 * and tempering are done simultaneously.
 * Resulted outputs are distributed in the range [0, 1).
 *
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @param[in] bid block id.
 * @return the tempered and converted value.
 */
__device__ float temper_single01(uint32_t V, uint32_t T, int bid) {
    uint32_t MAT;
    uint32_t r;

    T ^= T >> 16;
    T ^= T >> 8;
    MAT = tex1Dfetch(tex_single_ref, bid * 16 + (T & 0x0f));
    r = (V >> 9) ^ MAT;
    return __int_as_float(r) - 1.0f;
}

/**
 * Read the internal state vector from kernel I/O data, and
 * put them into shared memory.
 *
 * @param[out] status shared memory.
 * @param[in] d_status kernel I/O data
 * @param[in] bid block id
 * @param[in] tid thread id
 */
__device__ void status_read(uint32_t status[LARGE_SIZE],
			    const mtgp32_kernel_status_t *d_status,
			    int bid,
			    int tid) {
    status[LARGE_SIZE - N + tid] = d_status[bid].status[tid];
    if (tid < N - THREAD_NUM) {
	status[LARGE_SIZE - N + THREAD_NUM + tid]
	    = d_status[bid].status[THREAD_NUM + tid];
    }
    __syncthreads();
}

/**
 * Read the internal state vector from shared memory, and
 * write them into kernel I/O data.
 *
 * @param[out] d_status kernel I/O data
 * @param[in] status shared memory.
 * @param[in] bid block id
 * @param[in] tid thread id
 */
__device__ void status_write(mtgp32_kernel_status_t *d_status,
			     const uint32_t status[LARGE_SIZE],
			     int bid,
			     int tid) {
    d_status[bid].status[tid] = status[LARGE_SIZE - N + tid];
    if (tid < N - THREAD_NUM) {
	d_status[bid].status[THREAD_NUM + tid]
	    = status[4 * THREAD_NUM - N + tid];
    }
    __syncthreads();
}

/**
 * kernel function.
 * This function generates 32-bit unsigned integers in d_data
 *
 * @params[in,out] d_status kernel I/O data
 * @params[out] d_data output
 * @params[in] size number of output data requested.
 */
__global__ void mtgp32_uint32_kernel(mtgp32_kernel_status_t* d_status,
				     uint32_t* d_data, int size) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int pos = pos_tbl[bid];
    uint32_t r;
    uint32_t o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, bid, tid);

    // main loop
    for (int i = 0; i < size; i += LARGE_SIZE) {
	r = para_rec(status[LARGE_SIZE - N + tid],
		 status[LARGE_SIZE - N + tid + 1],
		 status[LARGE_SIZE - N + tid + pos],
		 bid);
	status[tid] = r;
	o = temper(r, status[LARGE_SIZE - N + tid + pos - 1], bid);
	d_data[size * bid + i + tid] = o;
	__syncthreads();

	r = para_rec(status[(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
		     status[(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
		     status[(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
		     bid);
	status[tid + THREAD_NUM] = r;
	o = temper(r,
		   status[(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
		   bid);
	d_data[size * bid + THREAD_NUM + i + tid] = o;
	__syncthreads();

	r = para_rec(status[2 * THREAD_NUM - N + tid],
		     status[2 * THREAD_NUM - N + tid + 1],
		     status[2 * THREAD_NUM - N + tid + pos],
		     bid);
	status[tid + 2 * THREAD_NUM] = r;
	o = temper(r, status[tid + pos - 1 + 2 * THREAD_NUM - N], bid);
	d_data[size * bid + 2 * THREAD_NUM + i + tid] = o;
	__syncthreads();
    }
    // write back status for next call
    status_write(d_status, status, bid, tid);
}

/**
 * kernel function.
 * This function generates single precision floating point numbers in d_data.
 *
 * @params[in,out] d_status kernel I/O data
 * @params[out] d_data output. IEEE single precision format.
 * @params[in] size number of output data requested.
 */
__global__ void mtgp32_single_kernel(mtgp32_kernel_status_t* d_status,
				     float* d_data, int size)
{

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int pos = pos_tbl[bid];
    uint32_t r;
    float o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, bid, tid);

    // main loop
    for (int i = 0; i < size; i += LARGE_SIZE) {
	r = para_rec(status[LARGE_SIZE - N + tid],
		     status[LARGE_SIZE - N + tid + 1],
		     status[LARGE_SIZE - N + tid + pos],
		     bid);
	status[tid] = r;
	o = temper_single(r, status[LARGE_SIZE - N + tid + pos - 1], bid);
	d_data[size * bid + i + tid] = o;
	__syncthreads();
	r = para_rec(status[(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
		     status[(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
		     status[(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
		     bid);
	status[tid + THREAD_NUM] = r;
	o = temper_single(
	    r,
	    status[(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
	    bid);
	d_data[size * bid + THREAD_NUM + i + tid] = o;
	__syncthreads();
	r = para_rec(status[2 * THREAD_NUM - N + tid],
		     status[2 * THREAD_NUM - N + tid + 1],
		     status[2 * THREAD_NUM - N + tid + pos],
		     bid);
	status[tid + 2 * THREAD_NUM] = r;
	o = temper_single(r,
			  status[tid + pos - 1 + 2 * THREAD_NUM - N],
			  bid);
	d_data[size * bid + 2 * THREAD_NUM + i + tid] = o;
	__syncthreads();
    }
    // write back status for next call
    status_write(d_status, status, bid, tid);
}

/**
 * kernel function.
 * This function generates single precision floating point numbers in d_data.
 *
 * @params[in,out] d_status kernel I/O data
 * @params[out] d_data output. IEEE single precision format.
 * @params[in] size number of output data requested.
 */
__global__ void mtgp32_single01_kernel(mtgp32_kernel_status_t* d_status,
				       float* d_data, int size)
{

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int pos = pos_tbl[bid];
    uint32_t r;
    float o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, bid, tid);

    // main loop
    for (int i = 0; i < size; i += LARGE_SIZE) {
	r = para_rec(status[LARGE_SIZE - N + tid],
		     status[LARGE_SIZE - N + tid + 1],
		     status[LARGE_SIZE - N + tid + pos],
		     bid);
	status[tid] = r;
	o = temper_single01(r, status[LARGE_SIZE - N + tid + pos - 1], bid);
	d_data[size * bid + i + tid] = o;
	__syncthreads();
	r = para_rec(status[(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
		     status[(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
		     status[(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
		     bid);
	status[tid + THREAD_NUM] = r;
	o = temper_single01(
	    r,
	    status[(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
	    bid);
	d_data[size * bid + THREAD_NUM + i + tid] = o;
	__syncthreads();
	r = para_rec(status[2 * THREAD_NUM - N + tid],
		     status[2 * THREAD_NUM - N + tid + 1],
		     status[2 * THREAD_NUM - N + tid + pos],
		     bid);
	status[tid + 2 * THREAD_NUM] = r;
	o = temper_single01(r,
			  status[tid + pos - 1 + 2 * THREAD_NUM - N],
			  bid);
	d_data[size * bid + 2 * THREAD_NUM + i + tid] = o;
	__syncthreads();
    }
    // write back status for next call
    status_write(d_status, status, bid, tid);
}

/**
 * This function initializes kernel I/O data.
 * @param d_status output kernel I/O data.
 * @param params MTGP32 parameters. needed for the initialization.
 */
void make_kernel_data32(mtgp32_kernel_status_t * d_status,
			mtgp32_params_fast_t params[],
			int block_num)
{
    int i;
    mtgp32_kernel_status_t* h_status
	= (mtgp32_kernel_status_t *) malloc(
	    sizeof(mtgp32_kernel_status_t) * block_num);

    if (h_status == NULL) {
	printf("failure in allocating host memory for kernel I/O data.\n");
	exit(8);
    }
    for (i = 0; i < block_num; i++) {
	mtgp32_init_state(&(h_status[i].status[0]), &params[i], i + 1);
    }
#if defined(DEBUG)
    printf("h_status[0].status[0]:%08"PRIx32"\n", h_status[0].status[0]);
    printf("h_status[0].status[1]:%08"PRIx32"\n", h_status[0].status[1]);
    printf("h_status[0].status[2]:%08"PRIx32"\n", h_status[0].status[2]);
    printf("h_status[0].status[3]:%08"PRIx32"\n", h_status[0].status[3]);
#endif
    ccudaMemcpy(d_status, h_status,
		sizeof(mtgp32_kernel_status_t) * block_num,
		cudaMemcpyHostToDevice);
    free(h_status);
}

/**
 * This function sets constants in device memory.
 * @param params input, MTGP32 parameters.
 */
void make_constant_param(const mtgp32_params_fast_t params[],
		   int block_num) {
    const int size1 = sizeof(uint32_t) * block_num;
    uint32_t *h_pos_tbl;
    uint32_t *h_sh1_tbl;
    uint32_t *h_sh2_tbl;
    uint32_t *h_mask;
    h_pos_tbl = (uint32_t *)malloc(size1);
    h_sh1_tbl = (uint32_t *)malloc(size1);
    h_sh2_tbl = (uint32_t *)malloc(size1);
    h_mask = (uint32_t *)malloc(sizeof(uint32_t));
    if (h_pos_tbl == NULL
	|| h_sh1_tbl == NULL
	|| h_sh2_tbl == NULL
	|| h_mask == NULL
	) {
	printf("failure in allocating host memory for constant table.\n");
	exit(1);
    }
    h_mask[0] = params[0].mask;
    for (int i = 0; i < block_num; i++) {
	h_pos_tbl[i] = params[i].pos;
	h_sh1_tbl[i] = params[i].sh1;
	h_sh2_tbl[i] = params[i].sh2;
    }
    // copy from malloc area only
    ccudaMemcpyToSymbol(pos_tbl, h_pos_tbl, size1);
    ccudaMemcpyToSymbol(sh1_tbl, h_sh1_tbl, size1);
    ccudaMemcpyToSymbol(sh2_tbl, h_sh2_tbl, size1);
    ccudaMemcpyToSymbol(mask, h_mask, sizeof(uint32_t));
    free(h_pos_tbl);
    free(h_sh1_tbl);
    free(h_sh2_tbl);
    free(h_mask);
}

/**
 * This function sets texture lookup table.
 * @param params input, MTGP32 parameters.
 * @param d_texture_tbl device memory used for texture bind
 * @param block_num block number used for kernel call
 */
void make_texture(const mtgp32_params_fast_t params[],
		  uint32_t *d_texture_tbl[3],
		  int block_num) {
    const int count = block_num * TBL_SIZE;
    const int size = sizeof(uint32_t) * count;
    uint32_t *h_texture_tbl[3];
    int i, j;
    for (i = 0; i < 3; i++) {
	h_texture_tbl[i] = (uint32_t *)malloc(size);
	if (h_texture_tbl[i] == NULL) {
	    for (j = 0; j < i; j++) {
		free(h_texture_tbl[i]);
	    }
	    printf("failure in allocating host memory for constant table.\n");
	    exit(1);
	}
    }
    for (int i = 0; i < block_num; i++) {
	for (int j = 0; j < TBL_SIZE; j++) {
	    h_texture_tbl[0][i * TBL_SIZE + j] = params[i].tbl[j];
	    h_texture_tbl[1][i * TBL_SIZE + j] = params[i].tmp_tbl[j];
	    h_texture_tbl[2][i * TBL_SIZE + j] = params[i].flt_tmp_tbl[j];
	}
    }
    ccudaMemcpy(d_texture_tbl[0], h_texture_tbl[0], size,
	       cudaMemcpyHostToDevice);
    ccudaMemcpy(d_texture_tbl[1], h_texture_tbl[1], size,
	       cudaMemcpyHostToDevice);
    ccudaMemcpy(d_texture_tbl[2], h_texture_tbl[2], size,
	       cudaMemcpyHostToDevice);
    tex_param_ref.filterMode = cudaFilterModePoint;
    tex_temper_ref.filterMode = cudaFilterModePoint;
    tex_single_ref.filterMode = cudaFilterModePoint;
    ccudaBindTexture(0, tex_param_ref, d_texture_tbl[0], size);
    ccudaBindTexture(0, tex_temper_ref, d_texture_tbl[1], size);
    ccudaBindTexture(0, tex_single_ref, d_texture_tbl[2], size);
    free(h_texture_tbl[0]);
    free(h_texture_tbl[1]);
    free(h_texture_tbl[2]);
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param d_status kernel I/O data.
 * @param num_data number of data to be generated.
 */
void make_uint32_random(mtgp32_kernel_status_t* d_status,
			int num_data,
			int block_num) {
    uint32_t* d_data;
    uint32_t* h_data;
    cudaError_t e;
    float gputime;
    cudaEvent_t start;
    cudaEvent_t end;

    printf("generating 32-bit unsigned random numbers.\n");
    ccudaMalloc((void**)&d_data, sizeof(uint32_t) * num_data);
    ccudaEventCreate(&start);
    ccudaEventCreate(&end);
    h_data = (uint32_t *) malloc(sizeof(uint32_t) * num_data);
    if (h_data == NULL) {
	printf("failure in allocating host memory for output data.\n");
	exit(1);
    }
    ccudaEventRecord(start, 0);
    if (cudaGetLastError() != cudaSuccess) {
	printf("error has been occured before kernel call.\n");
	exit(1);
    }

    /* kernel call */
    mtgp32_uint32_kernel<<< block_num, THREAD_NUM>>>(
	d_status, d_data, num_data / block_num);
    cudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
	printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
	exit(1);
    }
    ccudaEventRecord(end, 0);
    ccudaEventSynchronize(end);

    ccudaMemcpy(h_data,
		d_data,
		sizeof(uint32_t) * num_data,
		cudaMemcpyDeviceToHost);
    ccudaEventElapsedTime(&gputime, start, end);
    print_uint32_array(h_data, num_data, block_num);
    printf("generated numbers: %d\n", num_data);
    printf("Processing time: %f (ms)\n", gputime);
    printf("Samples per second: %E \n", num_data / (gputime * 0.001));
    ccudaEventDestroy(start);
    ccudaEventDestroy(end);
    //free memories
    free(h_data);
    ccudaFree(d_data);
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param d_status kernel I/O data.
 * @param num_data number of data to be generated.
 */
void make_single_random(mtgp32_kernel_status_t* d_status,
			int num_data,
			int block_num) {
    float* d_data;
    float* h_data;
    cudaError_t e;
    float gputime;
    cudaEvent_t start;
    cudaEvent_t end;

    printf("generating single precision floating point random numbers.\n");
    ccudaMalloc((void**)&d_data, sizeof(float) * num_data);
    ccudaEventCreate(&start);
    ccudaEventCreate(&end);
    h_data = (float *) malloc(sizeof(float) * num_data);
    if (h_data == NULL) {
	printf("failure in allocating host memory for output data.\n");
	exit(1);
    }
    ccudaEventRecord(start, 0);
    if (cudaGetLastError() != cudaSuccess) {
	printf("error has been occured before kernel call.\n");
	exit(1);
    }

    /* kernel call */
    mtgp32_single_kernel<<< block_num, THREAD_NUM >>>(
	d_status, d_data, num_data / block_num);
    cudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
	printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
	exit(1);
    }
    ccudaEventRecord(end, 0);
    ccudaEventSynchronize(end);
    ccudaMemcpy(h_data,
		d_data,
		sizeof(uint32_t) * num_data,
		cudaMemcpyDeviceToHost);
    ccudaEventElapsedTime(&gputime, start, end);
    print_float_array(h_data, num_data, block_num);
    printf("generated numbers: %d\n", num_data);
    printf("Processing time: %f (ms)\n", gputime);
    printf("Samples per second: %E \n", num_data / (gputime * 0.001));
    ccudaEventDestroy(start);
    ccudaEventDestroy(end);
    //free memories
    free(h_data);
    ccudaFree(d_data);
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param d_status kernel I/O data.
 * @param num_data number of data to be generated.
 */
void make_single01_random(mtgp32_kernel_status_t* d_status,
			  int num_data,
			  int block_num) {
    float* d_data;
    float* h_data;
    cudaError_t e;
    float gputime;
    cudaEvent_t start;
    cudaEvent_t end;

    printf("generating single precision floating point random numbers.\n");
    ccudaMalloc((void**)&d_data, sizeof(float) * num_data);
    ccudaEventCreate(&start);
    ccudaEventCreate(&end);
    h_data = (float *) malloc(sizeof(float) * num_data);
    if (h_data == NULL) {
	printf("failure in allocating host memory for output data.\n");
	exit(1);
    }
    ccudaEventRecord(start, 0);
    if (cudaGetLastError() != cudaSuccess) {
	printf("error has been occured before kernel call.\n");
	exit(1);
    }

    /* kernel call */
    mtgp32_single01_kernel<<< block_num, THREAD_NUM >>>(
	d_status, d_data, num_data / block_num);
    cudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
	printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
	exit(1);
    }
    ccudaEventRecord(end, 0);
    ccudaEventSynchronize(end);
    ccudaMemcpy(h_data,
		d_data,
		sizeof(uint32_t) * num_data,
		cudaMemcpyDeviceToHost);
    ccudaEventElapsedTime(&gputime, start, end);
    print_float_array(h_data, num_data, block_num);
    printf("generated numbers: %d\n", num_data);
    printf("Processing time: %f (ms)\n", gputime);
    printf("Samples per second: %E \n", num_data / (gputime * 0.001));
    ccudaEventDestroy(start);
    ccudaEventDestroy(end);
    //free memories
    free(h_data);
    ccudaFree(d_data);
}

int main(int argc, char *argv[])
{
    // LARGE_SIZE is a multiple of 16
    int num_data = 10000000;
    int block_num;
    int num_unit;
    int r;
    mtgp32_kernel_status_t *d_status;
    uint32_t *d_texture[3];
    int device = 0;
    int mb, mp;

    ccudaSetDevice(device);

    if (argc >= 2) {
	errno = 0;
	block_num = strtol(argv[1], NULL, 10);
	if (errno) {
	    printf("%s number_of_block number_of_output\n", argv[0]);
	    return 1;
	}
	if (block_num < 1 || block_num > BLOCK_NUM_MAX) {
	    printf("%s block_num should be between 1 and %d\n",
		   argv[0], BLOCK_NUM_MAX);
	    return 1;
	}
	errno = 0;
	num_data = strtol(argv[2], NULL, 10);
	if (errno) {
	    printf("%s number_of_block number_of_output\n", argv[0]);
	    return 1;
	}
	argc -= 2;
	argv += 2;
    } else {
	printf("%s number_of_block number_of_output\n", argv[0]);
	block_num = get_suitable_block_num(device,
					   &mb,
					   &mp,
					   sizeof(uint32_t),
					   THREAD_NUM,
					   LARGE_SIZE);
	if (block_num <= 0) {
	    printf("can't calculate sutable number of blocks.\n");
	    return 1;
	}
	printf("the suitable number of blocks for device 0 "
	       "will be multiple of %d, or multiple of %d\n", block_num,
	       (mb - 1) * mp);
	return 1;
    }

    num_unit = LARGE_SIZE * block_num;
    ccudaMalloc((void**)&d_status,
		sizeof(mtgp32_kernel_status_t) * block_num);
    ccudaMalloc((void**)&d_texture[0],
		sizeof(uint32_t) * block_num * TBL_SIZE);
    ccudaMalloc((void**)&d_texture[1],
		sizeof(uint32_t) * block_num * TBL_SIZE);
    ccudaMalloc((void**)&d_texture[2],
		sizeof(uint32_t) * block_num * TBL_SIZE);
    r = num_data % num_unit;
    if (r != 0) {
	num_data = num_data + num_unit - r;
    }
    make_constant_param(MTGPDC_PARAM_TABLE, block_num);
    make_texture(MTGPDC_PARAM_TABLE, d_texture, block_num);
    make_kernel_data32(d_status, MTGPDC_PARAM_TABLE, block_num);
    make_uint32_random(d_status, num_data, block_num);
    make_single_random(d_status, num_data, block_num);
    make_single01_random(d_status, num_data, block_num);

    //finalize
    ccudaFree(d_status);
    ccudaFree(d_texture[0]);
    ccudaFree(d_texture[1]);
    ccudaFree(d_texture[2]);
}
