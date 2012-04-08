//: nvcc mm.cu -o mm

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

__global__ void mm_kernel(float *d_m, float *d_n, float *d_p, int size)
{
    const int row = threadIdx.y;
    const int col = threadIdx.x;
    float val = 0.0;
    for (int i = 0; i < size; ++i)
    {
        val += d_m[row * size + i] * d_n[i * size + col];
    }
    d_p[row * size + col] = val;
}

void multiply(float *m, float *n, float *p, int size)
{
    // Pointers on the device:
    float *d_m, *d_n, *d_p;

    const int nbytes = (size * size) * sizeof(float);

    cudaMalloc((void**)&d_m, nbytes);
    cudaMalloc((void**)&d_n, nbytes);
    cudaMalloc((void**)&d_p, nbytes);
    
    cudaMemcpy(d_m, m, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, n, nbytes, cudaMemcpyHostToDevice);

    dim3 dimGrid(1, 1);
    dim3 dimBlock(size, size);
    mm_kernel<<<dimGrid,dimBlock>>>(d_m, d_n, d_p, size);

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

