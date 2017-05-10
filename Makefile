CC=g++ -Wall -O3 -std=c++1y -g

SYCL=/home/anastasi/Documents/Development/triSYCL/include
SYCL_OPT= -DNDEBUG -DBOOST_DISABLE_ASSERTS -fpermissive
OMP= -fopenmp

all: test knn_opencl

test: clean knn_trisycl_opencl_ASYNC knn_trisycl_opencl_NOASYNC knn_trisycl_openmp_ASYNC knn_trisycl_openmp_NOASYNC

knn_trisycl_opencl_ASYNC: knn_trisycl_opencl_interop.cpp
	$(CC) $(SYCL_OPT) -DTRISYCL_OPENCL $(OMP) -I$(SYCL) $< -o $@ -lOpenCL
knn_trisycl_opencl_NOASYNC: knn_trisycl_opencl_interop.cpp
	$(CC) $(SYCL_OPT) -DTRISYCL_NO_ASYNC -DTRISYCL_OPENCL $(OMP) -I$(SYCL) $< -o $@ -lOpenCL

knn_trisycl_openmp_ASYNC: knn_trisycl_openmp.cpp
	$(CC) $(SYCL_OPT) $(OMP) -I$(SYCL) $< -o $@
knn_trisycl_openmp_NOASYNC: knn_trisycl_openmp.cpp
	$(CC) $(SYCL_OPT) -DTRISYCL_NO_ASYNC $(OMP) -I$(SYCL) $< -o $@

knn_opencl: knn_opencl.cpp
	$(CC) $< -o $@ -lOpenCL

knn_trisycl_openmp: knn_trisycl_openmp.cpp
	$(CC) $(SYCL_OPT) -fpermissive $(OMP) -I$(SYCL) -o $@

clean:
	rm -f knn_opencl *ASYNC
