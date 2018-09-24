default:mm_avx2 mm_avx512

CC = icc
CFLAGS = -O3 -ipo -qopenmp -std=c11 -align -finline-functions
CLIBS = -qopenmp -ipo

CPUCFLAGS = $(CFLAGS) -xAVX2
KNLCFLAGS = $(CFLAGS) -xMIC-AVX512 -DMIC

test.o:test.c mm.h
	$(CC) $(CPUCFLAGS) -c test.c -o test.o

mm_avx2.o:mm.c mm.h
	$(CC) $(CPUCFLAGS) -c mm.c -o mm_avx2.o

mm_avx2:test.o mm_avx2.o
	$(CC) $(CLIBS) test.o mm_avx2.o -o mm_avx2

mm_avx512.o:mm.c mm.h
	$(CC) $(KNLCFLAGS) -c mm.c -o mm_avx512.o

mm_avx512:test.o mm_avx512.o
	$(CC) $(CLIBS) test.o mm_avx512.o -o mm_avx512
