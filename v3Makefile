all:	lake

lake:	lake_mpi.c  lakegpu_mpi.cu
	nvcc -c lakegpu_mpi.cu -Xcompiler -O  -arch=sm_20
	nvcc -c lake_mpi.c -Xcompiler -O2  -I/usr/mpi/gcc/openmpi-1.5.4/include
	mpic++ lakegpu_mpi.o lake_mpi.o -o lake -lm -L/usr/local/cuda/lib64/ -lcudart

clean:
	rm -f lake
