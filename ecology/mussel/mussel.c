// clang -DHAVE_INLINE -O3 mussel.c -o mussel -lgsl -lgslcblas

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

int main(int argc, char *argv[])
{
    // Setup the random number generator:
    gsl_rng_env_setup();
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus2);
    const unsigned int seed = (argc == 2)? atof(argv[1]) : time(NULL);
    gsl_rng_set(rng, seed);

    // Parameters:
    const int T = 1000;
    const double A0 = 0.5;
    const double A1 = 0.025;
    const double A2 = 0.2;
    const int N = 1000;
    const int TWO_N = N * N;

    // Allocate memory for the matrix:
    int **m = (int**)malloc(N * sizeof(int*));
    int i;
    for (i = 0; i < N; ++i)
    {
        m[i] = (int*)malloc(N * sizeof(int));
    }

    // Fill the matrix:
    int x0, y0;
    for (x0 = 0; x0 < N; ++x0)
    {
        for (y0 = 0; y0 < N; ++y0)
        {
            m[x0][y0] = (gsl_rng_uniform(rng) < 0.5) ? 0 : 2;
        }
    }

    // Main loop
    int n, x, y, x1, y1, t, cell;
    for (t = 0; t < T; ++t)
    {
        // Loop around the lattice at random to update cells;
        for (cell = 0; cell < TWO_N; ++cell)
        {
            // Select at random the cell to update
            x0 = (int)(gsl_rng_uniform(rng) * N);
            y0 = (int)(gsl_rng_uniform(rng) * N);

            // Update the mussels
            // Disturbed cells automatically transformed into empty cells
            if (m[x0][y0] == 0)
            {
                m[x0][y0] = 1;
            }
            else if (m[x0][y0] == 1) // If the cell is empty, test if there is colonization
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

                        if (x != 0 && y != 0 && m[x1][y1] == 2)
                        {
                            ++n;
                        }
                    }
                }
                // Calculate if the status if the cell is changed
                if (gsl_rng_uniform(rng) < A2 * n * 0.125)
                {
                    m[x0][y0] = 2;
                }
            }
            else if (m[x0][y0] == 2)
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

                        if (x != 0 && y != 0 && m[x1][y1] == 0)
                        {
                            n = 1;
                        }
                    }
                }
                // Calculate if the status of the cell is changed
                if (gsl_rng_uniform(rng) < A0 * n * 0.125 + A1)
                {
                    m[x0][y0] = 0;
                }
            }
        }
    }

    char buffer[100];
    sprintf(buffer, "mussel-%u.txt", seed);
    FILE *out = fopen(buffer, "w");

    for (x0 = 0; x0 < N; ++x0)
    {
        for (y0 = 0; y0 < N; ++y0)
        {
            fprintf(out, "%d ", m[x0][y0]);
        }
        fprintf(out, "\n");
    }
    fclose(out);

    gsl_rng_free(rng);
    for (i = 0; i < N; ++i)
    {
        free(m[i]);
    }
    free(m);
    return EXIT_SUCCESS;
}

