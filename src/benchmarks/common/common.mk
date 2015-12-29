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
TARGET_CODE 	:= $(BUILD_PATH)/$(TARGET)_code

all:				$(TARGET_PROFILER) $(TARGET_TRACE) $(TARGET_CODE)

$(TARGET_PROFILER):	$(CU_FILES) $(C_FILES) $(CPP_FILES)
	rm -f *.o
ifneq ("$(CU_FILES)", "")
	$(NVCC) -c $(CU_FILES) $(NVCC_FLAGS)
endif
ifneq ("$(C_FILES)", "")
	$(C) -c $(C_FILES) $(C_FLAGS)
endif
ifneq ("$(CPP_FILES)", "")
	$(CXX) -c $(CPP_FILES) $(CXX_FLAGS)
endif
	$(CXX) -o $(TARGET_PROFILER) *.o $(CXX_FLAGS) -L$(CUDA_LIB_PATH) -Wl,-rpath=$(CUDA_LIB_PATH) -lcudart
	$(CXX) -o $(TARGET_SIM) *.o $(CXX_FLAGS) -L$(CUDA_LIB_PATH) -lcudart
	rm -f *.o

TRACE_GENERATOR	:= ../../common/TraceGenerator.cpp
MAIN_MACRO_FLAG	:= -Dmain=original_main
$(TARGET_TRACE):	$(CU_FILES) $(C_FILES) $(CPP_FILES) $(TRACE_GENERATOR)
	rm -f *.o
ifneq ("$(CU_FILES)", "")
	$(NVCC) -c $(CU_FILES) $(NVCC_FLAGS) $(MAIN_MACRO_FLAG)
endif
ifneq ("$(C_FILES)", "")
	$(C) -c $(C_FILES) $(C_FLAGS) $(MAIN_MACRO_FLAG)
endif
ifneq ("$(CPP_FILES)", "")
	$(CXX) -c $(CPP_FILES) $(CXX_FLAGS) $(MAIN_MACRO_FLAG)
endif
	$(CXX) -c $(TRACE_GENERATOR) $(CXX_FLAGS)
	$(CXX) -o $(TARGET_TRACE) *.o $(CXX_FLAGS) -L$(CUDA_LIB_PATH) `OcelotConfig -l`
	rm -f *.o

CODE_GENERATOR	:= ../../common/CodeGenerator.cpp
MAIN_MACRO_FLAG	:= -Dmain=original_main
$(TARGET_CODE):	$(CU_FILES) $(C_FILES) $(CPP_FILES) $(CODE_GENERATOR)
	rm -f *.o
ifneq ("$(CU_FILES)", "")
	$(NVCC) -c $(CU_FILES) $(NVCC_FLAGS) $(MAIN_MACRO_FLAG)
endif
ifneq ("$(C_FILES)", "")
	$(C) -c $(C_FILES) $(C_FLAGS) $(MAIN_MACRO_FLAG)
endif
ifneq ("$(CPP_FILES)", "")
	$(CXX) -c $(CPP_FILES) $(CXX_FLAGS) $(MAIN_MACRO_FLAG)
endif
	$(CXX) -c $(CODE_GENERATOR) $(CXX_FLAGS)
	$(CXX) -o $(TARGET_CODE) *.o $(CXX_FLAGS) -L$(CUDA_LIB_PATH) `OcelotConfig -l`
	rm -f *.o
