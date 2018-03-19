default:mm_cpu mm_knl

CC = icc
CFLAGS = -O3 -ipo -qopenmp -std=c11 -align -finline-functions
CLIBS = -qopenmp -ipo

CPUCFLAGS = $(CFLAGS) -xHOST
KNLCFLAGS = $(CFLAGS) -xMIC-AVX512 -DMIC

test.o:test.c mm.h
	$(CC) $(CPUCFLAGS) -c test.c -o test.o

mm_cpu.o:mm.c mm.h
	$(CC) $(CPUCFLAGS) -c mm.c -o mm_cpu.o

mm_cpu:test.o mm_cpu.o
	$(CC) $(CLIBS) test.o mm_cpu.o -o mm_cpu

mm_knl.o:mm.c mm.h
	$(CC) $(KNLCFLAGS) -c mm.c -o mm_knl.o

mm_knl:test.o mm_knl.o
	$(CC) $(CLIBS) test.o mm_knl.o -o mm_knl

