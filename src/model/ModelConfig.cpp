


#include <iostream>

#include "ModelConfig.h"

void ModelConfig::calculate_line_bits() {
    int i;
    int n, m;

    n = cache_line_size;

    if (n < 2) {
        std::cout<< "####  ModelConfig::calculate_line_bits: too small cache line size  ####" << std::endl;
        return;
    }

    i = 0;
    while (n > 1) {
        m = n >> 1;
        if (n != (m << 1)) {
            std::cout<< "####  ModelConfig::calculate_line_bits: cache line size is not always even  ####" << std::endl;
            return;
        }
        i ++;
        n = m;
    }

    this->cache_line_bits = i;
    return;
}

void ModelConfig::print() {
    std::cout << "########  ModelConfig::Print  ########" << std::endl;
    std::cout << "cache_line_size:    " << cache_line_size << std::endl;
    std::cout << "cache_way_size:     " << cache_way_size << std::endl;
    std::cout << "cache_set_size:     " << cache_set_size << std::endl;
    std::cout << std::endl;
    std::cout << "allocate_on_miss:   " << allocate_on_miss << std::endl;
    std::cout << "jam_instruction:    " << jam_instruction << std::endl;
    std::cout << std::endl;
    std::cout << "latency_type:       " << latency_type << std::endl;
    std::cout << "latency_mean:       " << latency_mean << std::endl;
    std::cout << "latency_dev:        " << latency_dev << std::endl;
    std::cout << std::endl;
    std::cout << "num_sms:            " << num_sms << std::endl;
    std::cout << "max_active_blocks:  " << max_active_blocks << std::endl;
    std::cout << "max_active_threads: " << max_active_threads << std::endl;
    std::cout << std::endl;
    std::cout << "num_running_threads:" << num_running_threads << std::endl;
    std::cout << std::endl;
    std::cout << "mapping_type:       " << mapping_type << std::endl;
    std::cout << "coalescing_type:    " << coalescing_type << std::endl;
    std::cout << std::endl;
    std::cout << "mshr_check:         " << mshr_check << std::endl;
    std::cout << "num_mshrs:          " << num_mshrs << std::endl;
    std::cout << std::endl;
    std::cout << "cache_line_bits:  " << cache_line_bits << std::endl;
    std::cout << "#############  Print End  ############" << std::endl;
}
