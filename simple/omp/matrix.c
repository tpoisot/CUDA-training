#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.h"

void Matrix_init(Matrix *m, unsigned int nrow, unsigned int ncol)
{
    m->nrow = nrow;
    m->ncol = ncol;
    m->M = (double**)malloc(nrow * sizeof(double*));

    unsigned int i = 0;
    for(; i < nrow; i++)
    {
        m->M[i] = (double*)malloc(ncol * sizeof(double));
    }
}

void Matrix_init_fill(Matrix* m, unsigned int nrow, unsigned int ncol, double x)
{
    Matrix_init(m, nrow, ncol);
    Matrix_fill(m, x);
}

void Matrix_print(const Matrix *m, FILE* out)
{
    unsigned int i, j;
    for(i = 0; i < m->nrow; ++i)
    {
        for(j = 0; j < m->ncol; ++j)
        {
            fprintf(out, "%.4f ", m->M[i][j]);
        }
        fprintf(out, "\n");
    }
}

void Matrix_fill(Matrix *m, double x)
{
    unsigned int i, j;
    const unsigned int ncol = m->ncol;
    for(i = 0; i < m->nrow; ++i)
    {
        for(j = 0; j < ncol; ++j)
        {
            m->M[i][j] = x;
        }
    }
}

void Matrix_fill_fun(Matrix *m, double f(int i, int j)) {
    unsigned int r, c;
    const unsigned int ncol = m->ncol;
    for(r = 0; r < m->nrow; ++r)
    {
        for(c = 0; c < ncol; ++c)
        {
            m->M[r][c] = f(r, c);
        }
    }
}

void Matrix_free(Matrix *m)
{
    unsigned int i = 0;
    for(; i < m->nrow; ++i)
    {
        free(m->M[i]);
    }
    free(m->M);
}
