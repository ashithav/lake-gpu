all: lake

lake: lakegpu.cu lake.cu
	nvcc lakegpu.cu lake.cu -o ./lake -Xcompiler -O2 -lm

clean: 
	rm -f lake
