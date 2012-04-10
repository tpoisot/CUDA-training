//: nvcc mm.cu -o mm

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

__global__ void mm_kernel(float *d_m, float *d_n, float *d_p, int size)
{
    const int row = blockIdx.y;
    const int col = blockIdx.x;
    float val = 0.0;
    for (int i = 0; i < size; ++i)
    {
        val += d_m[row * size + i] * d_n[i * size + col];
    }
    d_p[row * size + col] = val;
}

/*
 * En declarant une variable comme __shared
 * tous les threads dans un bloc peuvent se partager la valeur
 * donner la valeur dans 
 * 
 * __shared_int sum;
 * if (threadIdx.x == 0)
 * {
 *  sum = 0; // ne peut etre initialisee que par le premier, par de maniere globale
 * }
 * __syncthread();
 * atomicAdd(&sum, 5); // la memoire partagee n'est pas changee par plusieurs threads
 * __syncthread(); // on attend que les autres threads soient OK pour passer a la suite
 * 
 * atomicAdd est VRAIMENT lent
 */

void multiply(float *m, float *n, float *p, int size)
{
    // Pointers on the device:
    float *d_m, *d_n, *d_p;
    
    // Matrice allouee directement dans un vecteur unique
    // sinon un nombre enorme de cuda malloc
    const int nbytes = (size * size) * sizeof(float);

    cudaMalloc((void**)&d_m, nbytes);
    cudaMalloc((void**)&d_n, nbytes);
    cudaMalloc((void**)&d_p, nbytes);
    
    cudaMemcpy(d_m, m, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, n, nbytes, cudaMemcpyHostToDevice);

    // definit une grid
    dim3 dimGrid(1, 1);
    // definit un block de size by size
    dim3 dimBlock(size, size);
    // balance sur les GPU
    mm_kernel<<<dimBlock,dimGrid>>>(d_m, d_n, d_p, size);
    
    cudaMemcpy(p, d_p, nbytes, cudaMemcpyDeviceToHost);

    cudaFree(d_m);
    cudaFree(d_n);
    cudaFree(d_p);
}

void print_matrix(float *m, FILE *out, int size)
{
    // Print matrix:
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            fprintf(out, "%8.4f ", m[i * size + j]);
        }
        fprintf(out, "\n");
    }
}

int main(int argc, char **argv)
{
    const int size = (argc == 2)? atof(argv[1]) : 10;
    const int nbytes = (size * size) * sizeof(float);

    float *m = (float*)malloc(nbytes);
    float *n = (float*)malloc(nbytes);
    float *p = (float*)malloc(nbytes);

    // Fill matrices m & n:
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            m[i * size + j] = cos(j) * sqrt(i);
            n[i * size + j] = sin(j) + 0.1 * i;
        }
    }

    multiply(m, n, p, size);

    print_matrix(p, stdout, size);

    free(m);
    free(n);
    free(p);
    return EXIT_SUCCESS;
}

