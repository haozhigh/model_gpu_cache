CUDA_HOME		:= /usr/local/cuda-4.0
CUDA_LIB_PATH	:= $(CUDA_HOME)/lib64
NVCC			:= $(CUDA_HOME)/bin/nvcc
C				:= /usr/bin/gcc-4.4
CXX				:= /usr/bin/g++-4.4

INCLUDE_FLAGS	:= -I./ -I../../common/ -I/usr/local/include/ocelot/api/interface -I$(CUDA_HOME)/include
DEBUG_FLAG      := -g
NVCC_FLAGS		:= -O3 -m64 -arch=sm_20 $(INCLUDE_FLAGS) $(DEBUG_FLAG)
C_FLAGS			:= -O3 -m64 $(INCLUDE_FLAGS) $(DEBUG_FLAG)
CXX_FLAGS		:= -O3 -m64 -std=c++0x $(INCLUDE_FLAGS) $(DEBUG_FLAG)

BUILD_PATH		    := ../../../../build
TARGET_PROFILER	    := $(BUILD_PATH)/$(TARGET)_profiler
TARGET_SIM		    := $(BUILD_PATH)/$(TARGET)_sim
TARGET_TRACE	    := $(BUILD_PATH)/$(TARGET)_trace
TARGET_BASE_TRACE   := $(BUILD_PATH)/$(TARGET)_base_trace
TARGET_CODE			:= $(BUILD_PATH)/$(TARGET)_code

all:				$(TARGET_PROFILER) $(TARGET_SIM) $(TARGET_TRACE) $(TARGET_BASE_TRACE) $(TARGET_CODE)

####  Rules to comiple bench source files to obj files
C_OBJS			:= $(patsubst %.c,%.o,$(C_FILES))
CPP_OBJS		:= $(patsubst %.cpp,%.o,$(CPP_FILES))
CU_OBJS			:= $(patsubst %.cu,%.o,$(CU_FILES))
OBJS			:= $(C_OBJS) $(CPP_OBJS) $(CU_OBJS)

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


####  Rules to generate profiler version executables
$(TARGET_PROFILER):	$(OBJS)
	$(CXX) -o $(TARGET_PROFILER) $(OBJS) $(CXX_FLAGS) -L$(CUDA_LIB_PATH) -Wl,-rpath=$(CUDA_LIB_PATH) -lcudart

####  Rules to generate Sim version executables
$(TARGET_SIM):		$(OBJS)
	$(CXX) -o $(TARGET_SIM) $(OBJS) $(CXX_FLAGS) -L$(CUDA_LIB_PATH) -lcudart

####  Rules to compile bench source files to obj files with the MAIN MACRO on, for trace generating
CTRACE_OBJS		:= $(patsubst %.c,%.otrace,$(C_FILES))
CPPTRACE_OBJS	:= $(patsubst %.cpp,%.otrace,$(CPP_FILES))
CUTRACE_OBJS	:= $(patsubst %.cu,%.otrace,$(CU_FILES))
TRACE_OBJS		:= $(CTRACE_OBJS) $(CPPTRACE_OBJS) $(CUTRACE_OBJS)

MAIN_MACRO_FLAG	:= -Dmain=original_main
ifneq ($(CTRACE_OBJS),)
$(CTRACE_OBJS):		%.otrace:	%.c
	$(C) -c $< $(C_FLAGS) $(MAIN_MACRO_FLAG) -o $@
endif
ifneq ($(CPPTRACE_OBJS),)
$(CPPTRACE_OBJS):		%.otrace:	%.cpp
	$(CXX) -c $< $(CXX_FLAGS) $(MAIN_MACRO_FLAG) -o $@
endif
ifneq ($(CUTRACE_OBJS),)
$(CUTRACE_OBJS):		%.otrace:	%.cu
	$(NVCC) -c $< $(NVCC_FLAGS) $(MAIN_MACRO_FLAG) -o $@
endif

####  Rules to compile trace generator source files to obj files
GENERATOR_PATH			:= ../../../trace
GENERATOR_CPP_FILES		:= $(GENERATOR_PATH)/TraceGenerator.cpp $(GENERATOR_PATH)/MyLastLoad.cpp $(GENERATOR_PATH)/MyWarpAccess.cpp
GENERATOR_OBJS			:= $(patsubst %.cpp,%.o,$(GENERATOR_CPP_FILES))

$(GENERATOR_OBJS):	%.o:	%.cpp
	$(CXX) -c $< -o $@ $(CXX_FLAGS)

####  Rules to generate trace generator executables
$(TARGET_TRACE):	$(TRACE_OBJS) $(GENERATOR_OBJS)
	$(CXX) -o $(TARGET_TRACE) $(TRACE_OBJS) $(GENERATOR_OBJS) $(CXX_FLAGS) -L$(CUDA_LIB_PATH) `OcelotConfig -l`





####  Rules to compile basetrace generator source files to obj files
BASE_GENERATOR_PATH          := ../../../base_trace
BASE_GENERATOR_CPP_FILES     := $(BASE_GENERATOR_PATH)/TraceGenerator.cpp
BASE_GENERATOR_OBJS          := $(patsubst %.cpp,%.o,$(BASE_GENERATOR_CPP_FILES))

$(BASE_GENERATOR_OBJS): %.o:    %.cpp
	$(CXX) -c $< -o $@ $(CXX_FLAGS)

#### Rules to generate base generator executables
$(TARGET_BASE_TRACE):   $(TRACE_OBJS) $(BASE_GENERATOR_OBJS)
	$(CXX) -o $(TARGET_BASE_TRACE) $(TRACE_OBJS) $(BASE_GENERATOR_OBJS) $(CXX_FLAGS) -L$(CUDA_LIB_PATH) `OcelotConfig -l`






#### Rules to generate code generator source files to obj files
CODE_PATH		:= ../../../code
CODE_CPP_FILES	:= $(CODE_PATH)/TraceGenerator.cpp
CODE_OBJS		:= $(patsubst %.cpp,%.o,$(CODE_CPP_FILES))

$(CODE_OBJS):	%.o:	%.cpp
	$(CXX) -c $< -o $@ $(CXX_FLAGS)

#### Rules to generate code executables
$(TARGET_CODE):		$(TRACE_OBJS) $(CODE_OBJS)
	$(CXX) -o $(TARGET_CODE) $(TRACE_OBJS) $(CODE_OBJS) $(CXX_FLAGS) -L$(CUDA_LIB_PATH) `OcelotConfig -l`


clean:
	rm -f $(OBJS) $(TRACE_OBJS) $(GENERATOR_OBJS) $(BASE_GENERATOR_OBJS) $(CODE_OBJS)
	rm -f $(TARGET_PROFILER) $(TARGET_SIM) $(TARGET_TRACE) $(TARGET_BASE_TRACE) $(TARGET_CODE)
