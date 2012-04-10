// nvcc mussel.cu -o cuda-mussel -lgsl -lgslcblas -lcuda

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <curand_kernel.h>

#define T       1000
#define A0      0.5
#define A1      0.025
#define A2      0.2
#define N       1000
#define TWO_N   (N*N)
#define BLOCKS  N
#define THREADS N
#ifndef SEED
#define SEED    1234
#endif

typedef unsigned int uint;

__global__ void gpu_mussel(uint *m0, uint *m1, curandState *states)
{
    const int x0 = threadIdx.x;
    const int y0 = blockIdx.x;
    const uint tid = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(SEED, tid, 0, &states[tid]);

    // Main loop
    int n, x, y, x1, y1;

    // Update the mussels
    // Disturbed cells automatically transformed into empty cells
    if (m0[tid] == 0)
    {
        m1[tid] = 1;
    }
    else if (m0[tid] == 1) // If the cell is empty, test if there is colonization
    {
        // First calculate the number of neighbours
        n = 0;
        for (x = -1; x <= 1; ++x)
        {
            for (y = -1; y <= 1; ++y)
            {
                if (x0 + x == -1)
                {
                    x1 = N - 1;
                }
                else if (x0 + x == N)
                {
                    x1 = 0;
                }
                else
                {
                    x1 = x0 + x;
                }

                if (y0 + y == -1)
                {
                    y1 = N - 1;
                }
                else if (y0 + y == N)
                {
                    y1 = 0;
                }
                else
                {
                    y1 = y0 + y;
                }

                if (x != 0 && y != 0 && m0[tid] == 2)
                {
                    ++n;
                }
            }
        }
        // Calculate if the status if the cell is changed
        if (curand_uniform(&states[tid]) < A2 * n * 0.125)
        {
            m1[tid] = 2;
        }
    }
    else if (m0[tid] == 2)
    { // If the cell is occupied, test if there is disturbance
        n = 0;
        for (x = -1; x <= 1; x++)
        {
            for (y = -1; y <= 1; y++)
            {
                if (x0 + x == -1)
                {
                    x1 = N - 1;
                }
                else if (x0 + x == N)
                {
                    x1 = 0;
                }
                else
                {
                    x1 = x0 + x;
                }

                if (y0 + y == -1)
                {
                    y1 = N - 1;
                }
                else if (y0 + y == N)
                {
                    y1 = 0;
                }
                else
                {
                    y1 = y0 + y;
                }

                if (x != 0 && y != 0 && m0[tid] == 0)
                {
                    n = 1;
                }
            }
        }
        // Calculate if the status of the cell is changed
        if (curand_uniform(&states[tid]) < A0 * n * 0.125 + A1)
        {
            m1[tid] = 0;
        }
    }
}

int main(int argc, char *argv[])
{
    const uint nbytes = TWO_N * sizeof(uint);
    uint *h_m = (uint*)malloc(nbytes);
    uint *d_m0, *d_m1;

    // Setup the random number generator:
    gsl_rng_env_setup();
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus2);
    gsl_rng_set(rng, SEED);

    // Fill the matrix:
    int x0, y0;
    for (x0 = 0; x0 < N; ++x0)
    {
        for (y0 = 0; y0 < N; ++y0)
        {
            h_m[y0 + x0 * N] = ((float)gsl_rng_uniform(rng) < 0.5) ? 0 : 2;
        }
    }

    // States for CUDA's random number generator:
    curandState *devStates;

    cudaMalloc((void**)&d_m0, nbytes);
    cudaMalloc((void**)&d_m1, nbytes);
    cudaMalloc((void**)&devStates, THREADS * BLOCKS * sizeof(curandState));

    cudaMemcpy(d_m0, h_m, nbytes, cudaMemcpyHostToDevice);

    for (int t = 0; t < T; t += 2)
    {
        gpu_mussel<<<BLOCKS, THREADS>>>(d_m0, d_m1, devStates);
        gpu_mussel<<<BLOCKS, THREADS>>>(d_m1, d_m0, devStates);
    }

    cudaMemcpy(h_m, d_m1, nbytes, cudaMemcpyDeviceToHost);

    char buffer[100];
    sprintf(buffer, "cuda-mussel-%u.txt", SEED);
    FILE *out = fopen(buffer, "w");

    for (x0 = 0; x0 < N; ++x0)
    {
        for (y0 = 0; y0 < N; ++y0)
        {
            fprintf(out, "%d ", h_m[y0 + x0 * N]);
        }
        fprintf(out, "\n");
    }
    fclose(out);

    gsl_rng_free(rng);
    free(h_m);
    cudaFree(d_m0);
    cudaFree(d_m1);
    return EXIT_SUCCESS;
}

