#include <string.h>
#include <omp.h>
#include "mm.h"

#define ALIGNED __attribute__((aligned(64)))

inline size_t min2(const size_t a, const size_t b)
{
    return (a < b) ? a : b;
}

#define MBLOCK 224
#define NBLOCK 48
#define KBLOCK (2 * MBLOCK)

inline void mm_serial(const size_t m, const size_t n, const size_t k,
                      const csmat_t a, const size_t lda, const csmat_t b, const size_t ldb, const smat_t c, const size_t ldc)
{
    size_t i, j, p, ibstart, jbstart, pbstart;
    ALIGNED data_t c_local[NBLOCK][MBLOCK];
#ifdef MIC
    ALIGNED data_t a_local[KBLOCK][MBLOCK];

    for (j = 0; j < n; ++j)
    {
        memset(c + CIDX(0, j, ldc), 0, sizeof(data_t) * m);
    }
    for (pbstart = 0; pbstart < k; pbstart += KBLOCK)
    {
        const size_t pbsize = min2(KBLOCK, k - pbstart);
        for (ibstart = 0; ibstart < m; ibstart += MBLOCK)
        {
            const size_t ibsize = min2(MBLOCK, m - ibstart);

            const cmat_t ab = a + CIDX(ibstart, pbstart, lda);
            for (p = 0; p < pbsize; ++p)
            {
                memcpy(a_local[p], ab + CIDX(0, p, lda), ibsize * sizeof(data_t));
            }

            for (jbstart = 0; jbstart < n; jbstart += NBLOCK)
            {
                const size_t jbsize = min2(NBLOCK, n - jbstart);

                const cmat_t bb = b + CIDX(pbstart, jbstart, ldb);
                const mat_t cb = c + CIDX(ibstart, jbstart, ldc);
                for (j = 0; j < jbsize; ++j)
                {
                    memset(c_local[j], 0, ibsize * sizeof(data_t));
                }
                for (j = 0; j < jbsize; ++j)
                {
                    for (p = 0; p < pbsize; ++p)
                    {
#pragma omp simd
                        for (i = 0; i < ibsize; ++i)
                        {
                            c_local[j][i] += a_local[p][i] * bb[CIDX(p, j, ldb)];
                        }
                    }
                }
                for (j = 0; j < jbsize; ++j)
                {
#pragma omp simd
                    for (i = 0; i < ibsize; ++i)
                    {
                        cb[CIDX(i, j, ldc)] += c_local[j][i];
                    }
                }
            }
        }
    }
#else
    ALIGNED data_t b_local[NBLOCK][KBLOCK];

    for (j = 0; j < n; ++j)
    {
        memset(c + CIDX(0, j, ldc), 0, sizeof(data_t) * m);
    }
    for (pbstart = 0; pbstart < k; pbstart += KBLOCK)
    {
        const size_t pbsize = min2(KBLOCK, k - pbstart);

        for (jbstart = 0; jbstart < n; jbstart += NBLOCK)
        {
            const size_t jbsize = min2(NBLOCK, n - jbstart);

            const cmat_t bb = b + CIDX(pbstart, jbstart, ldb);
            for (j = 0; j < jbsize; ++j)
            {
                memcpy(b_local[j], bb + CIDX(0, j, ldb), pbsize * sizeof(data_t));
            }
            for (ibstart = 0; ibstart < m; ibstart += MBLOCK)
            {
                const size_t ibsize = min2(MBLOCK, m - ibstart);

                const cmat_t ab = a + CIDX(ibstart, pbstart, lda);
                const mat_t cb = c + CIDX(ibstart, jbstart, ldc);

                for (j = 0; j < jbsize; ++j)
                {
                    memset(c_local[j], 0, ibsize * sizeof(data_t));
                }
                for (j = 0; j < jbsize; ++j)
                {
                    for (p = 0; p < pbsize; ++p)
                    {
#pragma omp simd
                        for (i = 0; i < ibsize; ++i)
                        {
                            c_local[j][i] += ab[CIDX(i, p, lda)] * b_local[j][p];
                        }
                    }
                }
                for (j = 0; j < jbsize; ++j)
                {
#pragma omp simd
                    for (i = 0; i < ibsize; ++i)
                    {
                        cb[CIDX(i, j, ldc)] += c_local[j][i];
                    }
                }
            }
        }
    }
#endif
}

inline size_t ceiling(const size_t den, const size_t num)
{
    return (den - 1) / num + 1;
}

static size_t partition_of_rows = 4;
static size_t partition_of_columns = 6;

void mm_set_partition(const size_t pr, const size_t pc)
{
    omp_set_num_threads(pr * pc);
    partition_of_rows = pr;
    partition_of_columns = pc;
}

void matrix_multiply(const size_t m, const size_t n, const size_t k,
                     const csmat_t a, const size_t lda,
                     const csmat_t b, const size_t ldb,
                     const smat_t c, const size_t ldc)
{
    const size_t pr = partition_of_rows;
    const size_t pc = partition_of_columns;
    const size_t rbsize_0 = ceiling(m, pr);
    const size_t cbsize_0 = ceiling(n, pc);
#pragma omp parallel shared(m, n, k, a, b, c, pr, pc, lda, ldb, ldc, rbsize_0, cbsize_0)
    {
        const size_t tid = (size_t)omp_get_thread_num();
        const size_t rid = tid % pr;
        const size_t cid = tid / pr;
        const size_t rbstart = rid * rbsize_0;
        const size_t cbstart = cid * cbsize_0;
        const size_t rbsize = min2(m - rbstart, rbsize_0);
        const size_t cbsize = min2(n - cbstart, cbsize_0);

        mm_serial(rbsize, cbsize, k,
                  a + CIDX(rbstart, 0, lda), lda,
                  b + CIDX(0, cbstart, ldb), ldb,
                  c + CIDX(rbstart, cbstart, ldc), ldc);
    }
}
