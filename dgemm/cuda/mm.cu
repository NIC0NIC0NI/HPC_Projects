#include <string.h>
#include <stdint.h>
#include <cuda.h>
#include "mm.h"

typedef uint32_t cusize_t;

__device__ __forceinline__ cusize_t min2(const cusize_t a, const cusize_t b)
{
    return (a < b) ? a : b;
}

#define RX 4
#define RY 4
#define BM 64
#define BN 64
#define BK 16
#define BDIMX 16
#define BDIMY 16

__global__ void mm_kernel(const cusize_t m, const cusize_t n, const cusize_t k,
                          const csmat_t a, const cusize_t lda,
                          const csmat_t b, const cusize_t ldb,
                          const smat_t c, const cusize_t ldc)
{
    const cusize_t ib = blockIdx.x * BM;
    const cusize_t jb = blockIdx.y * BN;
    const cusize_t ibrs = m - ib;
    const cusize_t jbrs = n - jb;
    const cusize_t i0 = threadIdx.x;
    const cusize_t i1 = threadIdx.x + BDIMX;
    const cusize_t i2 = threadIdx.x + BDIMX * 2;
    const cusize_t i3 = threadIdx.x + BDIMX * 3;
    const cusize_t j0 = threadIdx.y;
    const cusize_t j1 = threadIdx.y + BDIMY;
    const cusize_t j2 = threadIdx.y + BDIMY * 2;
    const cusize_t j3 = threadIdx.y + BDIMY * 3;
    data_t rC00 = 0.0, rC01 = 0.0, rC02 = 0.0, rC03 = 0.0;
    data_t rC10 = 0.0, rC11 = 0.0, rC12 = 0.0, rC13 = 0.0;
    data_t rC20 = 0.0, rC21 = 0.0, rC22 = 0.0, rC23 = 0.0;
    data_t rC30 = 0.0, rC31 = 0.0, rC32 = 0.0, rC33 = 0.0;
    data_t rA0, rA1, rA2, rA3, rB0, rB1, rB2, rB3;
    __shared__ data_t smA[BK][BM], smB[BK][BN];

    for (cusize_t pb = 0; pb < k; pb += BK)
    {
        const cusize_t pbsize = min2(BK, k - pb);
        for (cusize_t p = threadIdx.y; p < pbsize; p += BDIMY)
        {
            smA[p][i0] = (i0 < ibrs) ? a[CIDX(i0 + ib, p + pb, lda)] : 0.0;
            smA[p][i1] = (i1 < ibrs) ? a[CIDX(i1 + ib, p + pb, lda)] : 0.0;
            smA[p][i2] = (i2 < ibrs) ? a[CIDX(i2 + ib, p + pb, lda)] : 0.0;
            smA[p][i3] = (i3 < ibrs) ? a[CIDX(i3 + ib, p + pb, lda)] : 0.0;
            smB[p][i0] = (i0 < jbrs) ? b[CIDX(p + pb, i0 + jb, ldb)] : 0.0;
            smB[p][i1] = (i1 < jbrs) ? b[CIDX(p + pb, i1 + jb, ldb)] : 0.0;
            smB[p][i2] = (i2 < jbrs) ? b[CIDX(p + pb, i2 + jb, ldb)] : 0.0;
            smB[p][i3] = (i3 < jbrs) ? b[CIDX(p + pb, i3 + jb, ldb)] : 0.0;
        }

        __syncthreads();

        for (cusize_t p = 0; p < pbsize; ++p)
        {
            rA0 = smA[p][i0];
            rA1 = smA[p][i1];
            rA2 = smA[p][i2];
            rA3 = smA[p][i3];
            rB0 = smB[p][j0];
            rB1 = smB[p][j1];
            rB2 = smB[p][j2];
            rB3 = smB[p][j3];
            rC00 += rA0 * rB0;
            rC01 += rA0 * rB1;
            rC10 += rA1 * rB0;
            rC11 += rA1 * rB1;
            rC02 += rA0 * rB2;
            rC03 += rA0 * rB3;
            rC12 += rA1 * rB2;
            rC13 += rA1 * rB3;
            rC20 += rA2 * rB0;
            rC21 += rA2 * rB1;
            rC30 += rA3 * rB0;
            rC31 += rA3 * rB1;
            rC22 += rA2 * rB2;
            rC23 += rA2 * rB3;
            rC32 += rA3 * rB2;
            rC33 += rA3 * rB3;
        }

        __syncthreads();
    }
#define STORE(ii, jj) c[CIDX(i##ii + ib, j##jj + jb, ldc)] = rC##ii##jj;
#define CSTORE(ii, jj)                \
    if (i##ii < ibrs && j##jj < jbrs) \
    {                                 \
        STORE(ii, jj)                 \
    }
    if (ibrs >= BM && jbrs >= BN)
    {
        STORE(0, 0)
        STORE(0, 1)
        STORE(0, 2)
        STORE(0, 3)
        STORE(1, 0)
        STORE(1, 1)
        STORE(1, 2)
        STORE(1, 3)
        STORE(2, 0)
        STORE(2, 1)
        STORE(2, 2)
        STORE(2, 3)
        STORE(3, 0)
        STORE(3, 1)
        STORE(3, 2)
        STORE(3, 3)
    }
    else
    {
        CSTORE(0, 0)
        CSTORE(0, 1)
        CSTORE(0, 2)
        CSTORE(0, 3)
        CSTORE(1, 0)
        CSTORE(1, 1)
        CSTORE(1, 2)
        CSTORE(1, 3)
        CSTORE(2, 0)
        CSTORE(2, 1)
        CSTORE(2, 2)
        CSTORE(2, 3)
        CSTORE(3, 0)
        CSTORE(3, 1)
        CSTORE(3, 2)
        CSTORE(3, 3)
    }
#undef STORE
#undef CSTORE
}

inline size_t ceiling(const size_t den, const size_t num)
{
    return (den - 1) / num + 1;
}

void matrix_multiply_cuda(const size_t m, const size_t n, const size_t k,
                          const csmat_t a, const size_t lda,
                          const csmat_t b, const size_t ldb,
                          const smat_t c, const size_t ldc)
{
    mm_kernel<<<
        dim3(ceiling(m, BM), ceiling(n, BN), 1),
        dim3(BDIMX, BDIMY, 1)>>>(m, n, k, a, lda, b, ldb, c, ldc);
}
