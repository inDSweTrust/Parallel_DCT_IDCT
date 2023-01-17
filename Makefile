CC = g++

default: compression_omp

compression_omp: compression_omp.cpp
	${CC} -O0 -g -Wall -Wextra -Wno-unused-parameter -fopenmp -o $@ compression_omp.cpp -Lcnpy/ -lcnpy -lz --std=c++11

clean:
	-rm -f compression_omp

all: clean compression_omp