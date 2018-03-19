#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "mm.h"

typedef struct timespec timepoint_t;
#define GET_TIME(t) timespec_get(&t, TIME_UTC)
#define TIME_DIFF(start, end) ((double)(end.tv_sec - start.tv_sec) + 1e-9 * (double)(end.tv_nsec - start.tv_nsec))

void print_help(cstr_t argv0)
{
    printf("USAGE:\n");
    printf("\t%s <m> <n> <l> <row partition> <column partition>\n", argv0);
}

data_t square_error(const size_t n, const csmat_t c, const csmat_t c_ref)
{
    size_t i;
    data_t res = 0.0;
    for (i = 0; i < n; ++i)
    {
        data_t diff = c[i] - c_ref[i];
        res += diff * diff;
    }
    return res;
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
    for (i = 0; i < count; ++i){
        mat[i] = (data_t)rand() / (data_t)RAND_MAX;
    }
    return mat;
}

int main(int argc, args_t argv)
{
    timepoint_t t1, t2;
    size_t m, n, k, pr, pc;
    mat_t a, b, c;
    int im, in, ik, ipr, ipc;
    double t;

    srand(time(NULL));
    if (argc >= 6)
    {
        im = atoi(argv[1]);
        in = atoi(argv[2]);
        ik = atoi(argv[3]);
        ipr = atoi(argv[4]);
        ipc = atoi(argv[5]);
        if (im > 0 && in > 0 && ik > 0 && ipr > 0 && ipc > 0)
        {
            m = (size_t)im;
            n = (size_t)in;
            k = (size_t)ik;
            pr = (size_t)ipr;
            pc = (size_t)ipc;
            a = allocate_random(m * k);
            if (a != NULL)
            {
                b = allocate_random(k * n);
                if (b != NULL)
                {
                    c = allocate(m * n);
                    if (c != NULL)
                    {
                        mm_set_partition(pr, pc);

                        GET_TIME(t1);
                        matrix_multiply(m, n, k, a, m, b, k, c, m);
                        GET_TIME(t2);

                        t = TIME_DIFF(t1, t2);
                        printf("Time:\t\t\t%lfs\n", t);

                        deallocate(c);
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
