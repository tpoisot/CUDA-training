//: nvcc add1.cu -o add1

/*
 * Meme programme que add0 MAIS
 * on add les elements des vecteurs entre eux
 */

#include <stdlib.h>
#include <stdio.h>

// definition de la taille des vecteurs
#ifndef N
#define N 100
#endif

__global__ void add(int *a, int *b, int *c)
{
    int tid = blockIdx.x;
    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

int main(int argc, char **argv)
{
    // Memory on the host:
    int *h_a = (int*)malloc(N * sizeof(int));
    int *h_b = (int*)malloc(N * sizeof(int));
    int *h_c = (int*)malloc(N * sizeof(int));

    // Memory on the device:
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    // Fill the arrays:
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = i;
        h_b[i] = 2 * i * i - 6 * i;
    }

    // Copy from the host to the device:
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Call CUDA add:
    add<<<N,1>>>(d_a, d_b, d_c);
    /*
     * Signification du <<<N,1>>>
     * N : nombre de blocs
     * Pour chaque bloc, blockIdx.n, n est le numero du bloc
     * N doit etre inferieur a 65000 (verifier la taille exacte) 
     */

    // Copy from the device to the host:
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

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

