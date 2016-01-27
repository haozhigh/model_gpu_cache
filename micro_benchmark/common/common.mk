NVCC			= /usr/local/cuda-4.0/bin/nvcc

INCLUDE_FLAGS	:= -I./ -I../common/ -I/usr/local/cuda-4.0/include
NVCC_FLAGS		:= -arch=sm_20 $(INCLUDE_FLAGS)
CXXFLAGS		:= -Wall -O3 -m64
PTXASFLAGS		:= -v
CUDALIBPATH		= /usr/local/cuda-4.0/lib64
LINKERFLAGS		:= -rpath=${CUDALIBPATH}

all:			
				${NVCC} ${CUFILES} ${NVCCFLAGS} -o ${NAME} --ptxas-options ${PTXASFLAGS} --compiler-options ${CXXFLAGS} --linker-options -rpath=${CUDALIBPATH}

clean:			
				rm ${NAME}
