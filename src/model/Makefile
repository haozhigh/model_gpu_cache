CXX				:= /usr/bin/g++-4.8

INCLUDE_FLAGS	:= -I./
DEBUG_FLAG      := -g
CXX_FLAGS		:= -O3 -m64 -std=c++0x $(INCLUDE_FLAGS) $(DEBUG_FLAG)

CPP_FILES       := Access.cpp AnalyseTask.cpp ModelConfig.cpp DistanceStat.cpp functions.cpp io.cpp main.cpp NormalGenerator.cpp reuse_distance.cpp
CPP_OBJS		:= $(patsubst %.cpp,%.o,$(CPP_FILES))

BUILD_PATH		:= ../../build
TARGET          := $(BUILD_PATH)/model

TARGET:         $(CPP_OBJS)
				$(CXX) $(CXX_FLAGS) $(CPP_OBJS) -o $(TARGET) -lpthread

$(CPP_OBJS):	%.o:	%.cpp
				$(CXX) -c $< $(CXX_FLAGS) -o $@

clean:
				rm -f $(TARGET) $(CPP_OBJS)
