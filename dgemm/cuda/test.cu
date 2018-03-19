#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include "mm.h"

typedef struct timespec timepoint_t;
#define GET_TIME(t) timespec_get(&t, TIME_UTC)
#define TIME_DIFF(start, end) ((double)(end.tv_sec - start.tv_sec) + 1e-9 * (double)(end.tv_nsec - start.tv_nsec))

void print_help(cstr_t argv0)
{
    printf("USAGE:\n");
    printf("\t%s <m> <n> <l>\n", argv0);
}

mat_t allocate(const size_t count)
{
    return (mat_t)malloc(sizeof(data_t) * count);
}

void deallocate(const mat_t mat)
{
    free(mat);
}

mat_t allocate_random(const size_t count)
{
    mat_t mat = allocate(count);
    size_t i;
    for (i = 0; i < count; ++i)
    {
        mat[i] = (data_t)rand() / (data_t)RAND_MAX;
    }
    return mat;
}

mat_t allocate_gpu(const size_t count)
{
    mat_t a_dev;
    if (cudaMalloc(&a_dev, count * sizeof(data_t)) != cudaSuccess)
    {
        return NULL;
    }
    else
    {
        return a_dev;
    }
}

void deallocate_gpu(const mat_t mat)
{
    cudaFree(mat);
}

mat_t allocate_copy(const cmat_t a_host, const size_t count)
{
    mat_t a_dev = allocate_gpu(count);
    if (a_dev != NULL)
    {
        if (cudaMemcpy(a_dev, a_host, count * sizeof(data_t), cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cudaFree(a_dev);
            return NULL;
        }
    }
    return a_dev;
}

int main(int argc, args_t argv)
{
    timepoint_t t1, t2;
    size_t m, n, k;
    mat_t a, b, c_dev, a_dev, b_dev;
    int im, in, ik;
    double t;

    srand(time(NULL));
    if (argc >= 4)
    {
        im = atoi(argv[1]);
        in = atoi(argv[2]);
        ik = atoi(argv[3]);
        if (im > 0 && in > 0 && ik > 0)
        {
            m = (size_t)im;
            n = (size_t)in;
            k = (size_t)ik;
            a = allocate_random(m * k);
            if (a != NULL)
            {
                b = allocate_random(k * n);
                if (b != NULL)
                {
                    a_dev = allocate_copy(a, m * k);
                    if (a_dev != NULL)
                    {
                        b_dev = allocate_copy(b, k * n);
                        if (b_dev != NULL)
                        {
                            c_dev = allocate_gpu(m * n);
                            if (c_dev != NULL)
                            {
                                GET_TIME(t1);
                                matrix_multiply_cuda(m, n, k, a_dev, m, b_dev, k, c_dev, m);
                                cudaDeviceSynchronize();
                                GET_TIME(t2);

                                t = TIME_DIFF(t1, t2);
                                printf("Time:\t\t\t%lfs\n", t);

                                deallocate_gpu(c_dev);
                            }
                            deallocate_gpu(b_dev);
                        }
                        deallocate_gpu(a_dev);
                    }
                    deallocate(b);
                }
                else
                    printf("Input error.\n");
                deallocate(a);
            }
            else
                printf("Input error.\n");
        }
        else
            print_help(argv[0]);
    }
    else
        print_help(argv[0]);
    return 0;
}
