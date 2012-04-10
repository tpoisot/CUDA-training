//: nvcc add2.cu -o add2

#include <stdlib.h>
#include <stdio.h>

#define N       1000000
#define BLOCKS  1000
#define THREADS 512  

/*
 * Syntaxe : <<<BLOCKS,THREADS>>>
 * Pour chaque block, creation de copies distinctes avec un threadIdx.n
 * BLOCK   : petit bout de memoire de 14 bytes qui peuvent etre partages dans les threads
 * THREAD  : sous-division des threads
 * Nombre limite de threads -- 512, max 1024
 */

// Possible aussi de se referer aux x et aux y

__global__ void add(int *a, int *b, int *c)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x; // blockDim.x := 1000
    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

int main(int argc, char **argv)
{
    // Memory on the host:
    const size_t nbytes = N * sizeof(int);
    int *h_a = (int*)malloc(nbytes);
    int *h_b = (int*)malloc(nbytes);
    int *h_c = (int*)malloc(nbytes);

    // Memory on the device:
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, nbytes);
    cudaMalloc((void**)&d_b, nbytes);
    cudaMalloc((void**)&d_c, nbytes);

    // Fill the arrays:
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = i;
        h_b[i] = 2 * i * i - 6 * i;
    }

    // Copy from the host to the device:
    cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nbytes, cudaMemcpyHostToDevice);

    // Call CUDA add:
    add<<<BLOCKS,THREADS>>>(d_a, d_b, d_c);

    // Copy from the device to the host:
    cudaMemcpy(h_c, d_c, nbytes, cudaMemcpyDeviceToHost);

    // Print the results:
    for (int i = 0; i < N; ++i)
    {
        printf("(+ %d %d) -> %d\n", h_a[i], h_b[i], h_c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return EXIT_SUCCESS;
}

