#ifndef MTGP_UTIL_CUH
#define MTGP_UTIL_CUH
/*
 * mtgp-util.h
 *
 * Some utility functions for Sample Programs
 *
 */
#include <stdint.h>
#include <inttypes.h>
//#include "test-tool.hpp"

int get_suitable_block_num(int device, int *max_block_num,
			   int *mp_num, int word_size,
			   int thread_num, int large_size);
void print_max_min(uint32_t data[], int size);
void print_float_array(const float array[], int size, int block);
void print_uint32_array(const uint32_t array[], int size, int block);
void print_double_array(const double array[], int size, int block);
void print_uint64_array(const uint64_t array[], int size, int block);

inline void exception_maker(cudaError rc, const char * funcname)
{
    using namespace std;
    if (rc != cudaSuccess) {
	const char * message = cudaGetErrorString(rc);
	fprintf(stderr, "In %s Error(%d):%s\n", funcname, rc, message);
	throw message;
    }
}

inline int ccudaGetDeviceCount(int * num)
{
    cudaError rc = cudaGetDeviceCount(num);
    exception_maker(rc, "ccudaGetDeviceCount");
    return cudaSuccess;
}

inline int ccudaSetDevice(int dev)
{
    cudaError rc = cudaSetDevice(dev);
    exception_maker(rc, "ccudaSetDevice");
    return cudaSuccess;
}

inline int ccudaMalloc(void **devPtr, size_t size)
{
    cudaError rc = cudaMalloc((void **)(void*)devPtr, size);
    exception_maker(rc, "ccudaMalloc");
    return cudaSuccess;
}

inline int ccudaFree(void *devPtr)
{
    cudaError rc = cudaFree(devPtr);
    exception_maker(rc, "ccudaFree");
    return cudaSuccess;
}

inline int ccudaMemcpy(void *dest, void *src, size_t size,
		      enum cudaMemcpyKind kind)
{
    cudaError rc = cudaMemcpy(dest, src, size, kind);
    exception_maker(rc, "ccudaMemcpy");
    return cudaSuccess;
}

inline int ccudaEventCreate(cudaEvent_t * event)
{
    cudaError rc = cudaEventCreate(event);
    exception_maker(rc, "ccudaEventCreate");
    return cudaSuccess;
}

inline int ccudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    cudaError rc = cudaEventRecord(event, stream);
    exception_maker(rc, "ccudaEventRecord");
    return cudaSuccess;
}

inline int ccudaEventSynchronize(cudaEvent_t event)
{
    cudaError rc = cudaEventSynchronize(event);
    exception_maker(rc, "ccudaEventSynchronize");
    return cudaSuccess;
}

inline int ccudaThreadSynchronize()
{
    cudaError rc = cudaThreadSynchronize();
    exception_maker(rc, "ccudaThreadSynchronize");
    return cudaSuccess;
}

inline int ccudaEventElapsedTime(float * ms,
				 cudaEvent_t start, cudaEvent_t end)
{
    cudaError rc = cudaEventElapsedTime(ms, start, end);
    exception_maker(rc, "ccudaEventElapsedTime");
    return cudaSuccess;
}

inline int ccudaEventDestroy(cudaEvent_t event)
{
    cudaError rc = cudaEventDestroy(event);
    exception_maker(rc, "ccudaEventDestroy");
    return cudaSuccess;
}

inline int ccudaMemcpyToSymbol(const void * symbol,
			       const void * src,
			       size_t count,
			       size_t offset = 0,
			       enum cudaMemcpyKind kind
			       = cudaMemcpyHostToDevice)
{
    cudaError rc = cudaMemcpyToSymbol((const char *)symbol,
					src, count, offset, kind);
    exception_maker(rc, "ccudaMemcpyToSymbol");
    return cudaSuccess;
}

inline int ccudaGetDeviceProperties(struct cudaDeviceProp * prop, int device)
{
    cudaError rc = cudaGetDeviceProperties(prop, device);
    exception_maker(rc, "ccudaGetDeviceProperties");
    return cudaSuccess;
}

template<class T, int dim, enum cudaTextureReadMode readMode>
inline int ccudaBindTexture(size_t * offset,
			    const struct texture< T, dim, readMode > & texref,
			    const void * devPtr,
			    size_t size = UINT_MAX)
{

    cudaError rc = cudaBindTexture(offset, texref, devPtr, size);
    exception_maker(rc, "ccudaBIndTexture");
    return cudaSuccess;
}

#endif
