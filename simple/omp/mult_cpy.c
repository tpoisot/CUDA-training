#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "Matrix.h"

#define CHUNKSIZE 16

// The Matrix struct;
typedef struct {
	double** M; // The actual matrix
	int nrow; // Number of rows
	int ncol; // Number of columns
} Matrix;

void Matrix_init(Matrix* m, int nrow, int ncol) {
	m->nrow = nrow;
	m->ncol = ncol;
	m->M = (double**)malloc(nrow * sizeof(double*));

	int i;
	for(i = 0; i < nrow; i++)
		m->M[i] = (double*)malloc(ncol * sizeof(double));
}

void Matrix_init_fill(Matrix* m, int nrow, int ncol, double x) {
	Matrix_init(m, nrow, ncol);
	Matrix_fill(m, x);
}

void Matrix_mult(const Matrix* a, const Matrix* b, Matrix* c) {
	assert(a->ncol == b->nrow);
	if(c->M != NULL)
		Matrix_free(c);
	Matrix_init_fill(c, a->nrow, b->ncol, 0.0);

	int i, j, k; // Index
	double bncolj[a->ncol]; // To copy arrays	

	#ifdef _OPENMP
	int	tid, nthreads, chunk = CHUNKSIZE;
	#pragma omp parallel shared(a, b, c, nthreads, chunk) private(tid, i, j, k, bncolj)
	{
		tid = omp_get_thread_num();
		if(tid == 0) {
			nthreads = omp_get_num_threads();
			// printf("Using OpenMP with %d threads\n", nthreads);
		}
		#pragma omp for schedule(static, chunk)
	#endif
		for(j = 0; j < b->ncol; j++) {
			for(k = 0; k < a->ncol; k++)
				bncolj[k] = b->M[k][j];

			for(i = 0; i < a->nrow; i++) { 
				const double *anrowi = a->M[i];
				double s = 0.0;   
				for(k = 0; k < a->ncol; k++)
					s += anrowi[k] * bncolj[k];
				c->M[i][j] = s;
			}
		}
	#ifdef _OPENMP
	} // End of parallel region
	#endif
}

void Matrix_print(const Matrix *m, FILE* out) {
	int i, j;
	if(out == NULL) {
		for(i = 0; i < m->nrow; i++) {
			for(j = 0; j < m->ncol; j++)
				printf("%.4f ", m->M[i][j]);
			printf("\n");
		}
	} else {
		for(i = 0; i < m->nrow; i++) {
			for(j = 0; j < m->ncol; j++)
				fprintf(out, "%.4f ", m->M[i][j]);
			fprintf(out, "\n");
		}
	}
}

void Matrix_fill(Matrix *m, double x) {
	int i, j;
	for(i = 0; i < m->nrow; i++)
		for(j = 0; j < m->ncol; j++)
			m->M[i][j] = x;
}

void Matrix_fill_fun(Matrix *m, double f(int i, int j)) {
	int ii, jj;
	for(ii = 0; ii < m->nrow; ii++)
		for(jj = 0; jj < m->ncol; jj++)
			m->M[ii][jj] = f(ii, jj);
}

void Matrix_free(Matrix *m) {
	int i;
	for(i = 0; i < m->nrow; i++)
		free(m->M[i]);
	free(m->M);
}
