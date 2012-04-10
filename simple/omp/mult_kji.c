/**
 * Exactly like mat1.c, but with kji order.
 * 
 * Written with Geany
 * compilation:
 *    gcc -Wall -O3 -o "Mat3" "Mat3.c"
 *
 * With N = 2000
 * Time = 122.75s
 * 
 * Computer: Dual Core 2.8ghz
 * OS: Fedora 11 64bits (Gnome)
 * Compiler: GCC 4.4.0
 *********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2000

/**
 * A structure for a matrix stored in a jagged array
 */
typedef struct {
    int row; /** Number of rows */
    int col; /** Number of columns */
    double **M; /** Pointer to the array */
} Mat3;

Mat3 * init_Mat3(int row, int col);
void fill_Mat3(Mat3 *m, double x);
void print_Mat3(Mat3 *m);
void Mat3_mult(Mat3 *a, Mat3 *b, Mat3 *c);

int main()
{
    Mat3 *a1 = init_Mat3(N, N);
    fill_Mat3(a1, 1.5);

    Mat3 *b1 = init_Mat3(N, N);
    fill_Mat3(b1, 0.25);

    Mat3 *c1 = init_Mat3(N, N);
    
    clock_t start = clock();
    
    int trials;
    for(trials = 0; trials < 1; trials++)
        Mat3_mult(a1, b1, c1);

    printf( "Time elapsed: %f\n", ( (double)clock() - start ) / CLOCKS_PER_SEC );

    return 0;
}

//
// MATRIX 1 FUNCTIONS
//

Mat3 * init_Mat3(int row, int col) {
    Mat3 *temp;
    temp = malloc( sizeof(Mat3) );

    (*temp).row = row;
    (*temp).col = col;
    (*temp).M = malloc(row * sizeof(double*));

    int i;
    for(i = 0; i < row; i++)
        (*temp).M[i] = malloc(col * sizeof(double));

    return temp;
}

void print_Mat3(Mat3 *m) {
    int i, j;
    for(i = 0; i < (*m).row; i++) {
        for(j = 0; j < (*m).col; j++)
            printf("%.4f ", (*m).M[i][j]);
        printf("\n");
    }
}

void fill_Mat3(Mat3 *m, double x) {
    int i, j;
    for(i = 0; i < (*m).row; i++)
        for(j = 0; j < (*m).col; j++)
            (*m).M[i][j] = x;
}

void Mat3_mult(Mat3 *a, Mat3 *b, Mat3 *c) {
    fill_Mat3(c, 0.00);

    int k, i, j;
    for(k = 0; k < N; k++)
        for(j = 0; j < N; j++)
            for(i = 0; i < N; i++)
                (*c).M[i][j] += (*a).M[i][k] * (*b).M[k][j];
}
