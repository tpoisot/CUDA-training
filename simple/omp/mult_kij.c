/**
 * kij order, with 2d arrays.
 * 
 * Written with Geany
 * compilation:
 *    gcc -Wall -O3 -o "mat1" "mat1.c"
 *
 * With N = 2000
 * Time = 28.28s
 * 
 * Computer: Dual Core 2.8ghz
 * OS: Fedora 11 64bits (Gnome)
 * Compiler: GCC 4.4.0
 *********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"

void Matrix_mult(const Matrix *a, const Matrix *b, Matrix *c)
{
	int k, i, j;
	for (k = 0; k < N; k++)
    {
		for (i = 0; i < N; i++)
        {
			for (j = 0; j < N; j++)
            {
				c->M[i][j] += a->M[i][k] * b->M[k][j];
            }
        }
    }
}

int main() {

		return 0;
}
