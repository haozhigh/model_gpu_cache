NVCC		= /usr/local/cuda-4.0/bin/nvcc
CXX			= g++

CXXFLAGS	= -m64 -std=c++0x -Wall
NVCCFLAGS	= -m64 --ptxas-options -dlcm=ca

CUDALIBPATH	= /usr/local/cuda-4.0/lib64

cache:
			${NVCC} -c ${NVCCFLAGS} cache1.cu -o cache1.o
			${CXX} cache1.o ${CXXFLAGS} -L${CUDALIBPATH} -lcudart -o cache1 -Wl,-rpath=${CUDALIBPATH} 
			rm -f cache1.o
clean:		
			rm -f cache1
