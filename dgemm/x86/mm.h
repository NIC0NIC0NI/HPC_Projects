#ifndef MM_H_INCLUDED
#define MM_H_INCLUDED

#include <stddef.h>

typedef const char *cstr_t;
typedef const cstr_t *args_t;
typedef double data_t;
typedef data_t* mat_t;
typedef const data_t* cmat_t;
typedef data_t* restrict smat_t;
typedef const data_t* restrict csmat_t;
/*
#define RIDX(i, j, ld) ((i) * (ld) + (j))
*/
// column major index
#define CIDX(i, j, ld) ((i) + (j) * (ld))



// Not thread safe
void mm_set_partition(const size_t pr, const size_t pc);

void matrix_multiply(const size_t m, const size_t n, const size_t k,
                         const csmat_t a, const size_t lda, 
                         const csmat_t b, const size_t ldb, 
                         const smat_t c, const size_t ldc);

#endif