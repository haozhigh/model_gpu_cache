NVCC		= /usr/local/cuda-4.0/bin/nvcc
CXX			= g++

CXXFLAGS	= -m64 -g -std=c++0x -Wall
NVCCFLAGS	= -m64 -g --ptxas-options -dlcm=ca

CUDALIBPATH	= /usr/local/cuda-4.0/lib64

main:
			${NVCC} -c ${NVCCFLAGS} main.cu -o main.o
			${CXX} main.o ${CXXFLAGS} -L${CUDALIBPATH} -lcudart -o main -Wl,-rpath=${CUDALIBPATH} 
			rm -f main.o
clean:		
			rm -f main
