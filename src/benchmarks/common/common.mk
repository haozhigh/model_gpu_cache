CUDA_HOME		:= /usr/local/cuda-4.0
CUDA_LIB_PATH	:= $(CUDA_HOME)/lib64
NVCC			:= $(CUDA_HOME)/bin/nvcc
C				:= /usr/bin/gcc-4.4
CXX				:= /usr/bin/g++-4.4

INCLUDE_FLAGS	:= -I./ -I../../common/ -I/usr/local/include/ocelot/api/interface -I/usr/local/cuda-4.0/include
DEBUG_FLAG      := -g
NVCC_FLAGS		:= -O3 -m64 -arch=sm_20 $(INCLUDE_FLAGS) $(DEBUG_FLAG)
C_FLAGS			:= -O3 -m64 $(INCLUDE_FLAGS) $(DEBUG_FLAG)
CXX_FLAGS		:= -O3 -m64 -std=c++0x $(INCLUDE_FLAGS) $(DEBUG_FLAG)

BUILD_PATH		:= ../../../../build
TARGET_PROFILER	:= $(BUILD_PATH)/$(TARGET)_profiler
TARGET_SIM		:= $(BUILD_PATH)/$(TARGET)_sim
TARGET_TRACE	:= $(BUILD_PATH)/$(TARGET)_trace

C_OBJS			:= $(patsubst %.c,%.o,$(C_FILES))
CPP_OBJS		:= $(patsubst %.cpp,%.o,$(CPP_FILES))
CU_OBJS			:= $(patsubst %.cu,%.o,$(CU_FILES))
OBJS			:= $(C_OBJS) $(CPP_OBJS) $(CU_OBJS)

all:				$(TARGET_PROFILER) $(TARGET_TRACE)

ifneq ($(C_OBJS),)
$(C_OBJS):	%.o:	%.c
	$(C) -c $< $(C_FLAGS) -o $@
endif
ifneq ($(CPP_OBJS),)
$(CPP_OBJS):	%.o:	%.cpp
	$(CXX) -c $< $(CXX_FLAGS) -o $@
endif
ifneq ($(CU_OBJS),)
$(CU_OBJS):	%.o:	%.cu
	$(NVCC) -c $< $(NVCC_FLAGS) -o $@
endif

$(TARGET_PROFILER):	$(OBJS)
	$(CXX) -o $(TARGET_PROFILER) $(OBJS) $(CXX_FLAGS) -L$(CUDA_LIB_PATH) -Wl,-rpath=$(CUDA_LIB_PATH) -lcudart

$(TARGET_SIM):		$(OBJS)
	$(CXX) -o $(TARGET_SIM) $(OBJS) $(CXX_FLAGS) -L$(CUDA_LIB_PATH) -lcudart

CTRACE_OBJS		:= $(patsubst %.c,%.otrace,$(C_FILES))
CPPTRACE_OBJS	:= $(patsubst %.cpp,%.otrace,$(CPP_FILES))
CUTRACE_OBJS	:= $(patsubst %.cu,%.otrace,$(CU_FILES))
TRACE_OBJS		:= $(CTRACE_OBJS) $(CPPTRACE_OBJS) $(CUTRACE_OBJS)

MAIN_MACRO_FLAG	:= -Dmain=original_main
ifneq ($(CTRACE_OBJS,))
$(CTRACE_OBJS):		%.otrace:	%.c
	$(C) -c $< $(C_FLAGS) $(MAIN_MACRO_FLAG) -o $@
endif
ifneq ($(CPPTRACE_OBJS,))
$(CPPTRACE_OBJS):		%.otrace:	%.cpp
	$(CXX) -c $< $(CXX_FLAGS) $(MAIN_MACRO_FLAG) -o $@
endif
ifneq ($(CUTRACE_OBJS,))
$(CUTRACE_OBJS):		%.otrace:	%.cu
	$(NVCC) -c $< $(NVCC_FLAGS) $(MAIN_MACRO_FLAG) -o $@
endif

GENERATOR_OBJS	:= ../../../trace/*.o
$(TARGET_TRACE):	$(TRACE_OBJS)
	$(CXX) -o $(TARGET_TRACE) $(TRACE_OBJS) $(CXX_FLAGS) -L$(CUDA_LIB_PATH) `OcelotConfig -l`


clean:
	rm $(OBJS) $(TRACE_OBJS)
