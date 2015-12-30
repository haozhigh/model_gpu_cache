


#include <iostream>

#include "ModelConfig.h"


void ModelConfig::print() {
    std::cout << "########  ModelConfig::Print  ########" << std::endl;
    std::cout << "cache_line_size:  " << cache_line_size << std::endl;
    std::cout << "cache_way_size:   " << cache_way_size << std::endl;
    std::cout << "cache_set_size:   " << cache_set_size << std::endl;
    std::cout << "allocate_on_miss: " << allocate_on_miss << std::endl;
    std::cout << "jam_instruction:  " << jam_instruction << std::endl;
    std::cout << "latency_type:     " << latency_type << std::endl;
    std::cout << "latency_mean:     " << latency_mean << std::endl;
    std::cout << "latency_dev:      " << latency_dev << std::endl;
    std::cout << "#############  Print End  ############" << std::endl;
}
