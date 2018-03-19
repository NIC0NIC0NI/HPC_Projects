default:mm

CUDAC = nvcc
CFLAGS = -O3 -Xptxas "-dlcm=cg -dscm=cs" --gpu-architecture=compute_35 --gpu-code=sm_35
CLIBS = 

test.o:test.cu mm.h
	$(CUDAC) $(CFLAGS) -c test.cu -o test.o

mm.o:mm.cu mm.h
	$(CUDAC) $(CFLAGS) -c mm.cu -o mm.o
	
mm:test.o mm.o
	$(CUDAC) $(CLIBS) test.o mm.o -o mm


