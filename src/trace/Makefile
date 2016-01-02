CUDA_HOME		:= /usr/local/cuda-4.0
CUDA_LIB_PATH	:= $(CUDA_HOME)/lib64
NVCC			:= $(CUDA_HOME)/bin/nvcc
C				:= /usr/bin/gcc-4.4
CXX				:= /usr/bin/g++-4.4

INCLUDE_FLAGS	:= -I./  -I/usr/local/include/ocelot/api/interface -I/usr/local/cuda-4.0/include
DEBUG_FLAG      := -g
NVCC_FLAGS		:= -O3 -m64 -arch=sm_20 $(INCLUDE_FLAGS) $(DEBUG_FLAG)
C_FLAGS			:= -O3 -m64 $(INCLUDE_FLAGS) $(DEBUG_FLAG)
CXX_FLAGS		:= -O3 -m64 -std=c++0x $(INCLUDE_FLAGS) $(DEBUG_FLAG)


CPP_FILES		:= TraceGenerator.cpp MyLastLoad.cpp MyWarpAccess.cpp
OBJS			:= $(patsubst %.cpp,%.o,$(CPP_FILES))

all:		$(OBJS)

$(OBJS):	%.o:	%.cpp
	$(CXX) -c $< -o $@ $(CXX_FLAGS)

clean:      
			rm $(OBJS)
