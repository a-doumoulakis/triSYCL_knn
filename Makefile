CC=g++ -Wall -std=c++1y

GCCL=/usr/lib/gcc/x86_64-linux-gnu/6.2.0/include
SYCL=/home/archon/Documents/triSYCL/include

all: knn_opencl knn_trisycl knn_pure_opencl

knn_opencl: knn_opencl.cpp
	$(CC) -DTRISYCL_OPENCL -I$(SYCL) -fopenmp $< -o $@ -lOpenCL

knn_trisycl: knn_trisycl.cpp
	$(CC) -I$(SYCL) -fopenmp $< -o $@

knn_pure_opencl: knn_pure_opencl.cpp
	$(CC) $< -o $@ -lOpenCL

clean:
	rm -f knn_trisycl knn_opencl knn_pure_opencl
