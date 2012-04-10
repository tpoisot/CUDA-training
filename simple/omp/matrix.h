#ifndef MATRIX_H_
#define MATRIX_H_

/** A dense matric of double. */
typedef struct
{
    double** M; /**< The actual matrix **/

    unsigned int nrow; /**< Number of rows **/

    unsigned int ncol; /**< Number of columns **/
}
Matrix;

/** Initialize the matrix. */
void Matrix_init(Matrix* m, unsigned int nrow, int ncol);

/** Initilize the matrix and fill it. */
void Matrix_init_fill(Matrix* m, unsigned int nrow, unsigned int ncol, double x);

/** Print the matrix. Give 'stdin' as argument to print to the console. **/
void Matrix_print(const Matrix *m, FILE* out);

/** Fill the entire matrix with some value. */
void Matrix_fill(Matrix *m, double x);

/** Fill with a function */
void Matrix_fill_fun(Matrix *m, double f(unsigned int i, unsigned int j));

/** Free memory. */
void Matrix_free(Matrix *m);

#endif