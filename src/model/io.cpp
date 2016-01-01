

#include <iostream>
#include <fstream>
#include <sstream>

#include "io.h"
#include "functions.h"

void read_model_config_from_file(std::string file_path, ModelConfig &model_config) {
    std::ifstream in_stream;
    
    //  Set default config
    model_config.cache_line_size = 128;
    model_config.cache_way_size = 4;
    model_config.cache_set_size = 32;

    model_config.allocate_on_miss = 1;
    model_config.jam_instruction = 1;

    model_config.latency_type = 0;
    model_config.latency_mean = 100;
    model_config.latency_dev = 5;

    in_stream.open(file_path, std::ifstream::in);
    //  If the config file can not be opened, use the default config
    if (! in_stream.is_open()) {
        std::cout<< "####  read_model_config_from_file: config file cannot be opened, use the default config  ####" << std::endl;
        return;
    }

    //  Read config from the stream
    std::string line;
    std::string str_item;
    std::string str_value;
    int value;
    while (std::getline(in_stream, line)) {
        int i;

        //  Split each line to two parts, and trim them individually
        i = line.find(' ');
        if (i < 0) {
            std::cout << "#### read_model_config_from_file: Unsupported config line: " << line << std::endl;
            continue;
        }
        str_item = line.substr(0, i);
        str_value = line.substr(i, std::string::npos);
        string_trim(str_item);
        string_trim(str_value);

        //  Make sure that the right part of each line is of int type
        if (! string_is_int(str_value)) {
            std::cout << "#### read_model_config_from_file: Unsupported config line: " << line << std::endl;
            continue;
        }
        value = string_to_int(str_value);

        //  Check for each possible config item
        if (str_item == "cache_line_size") {
            model_config.cache_line_size = value;
            continue;
        }

        if (str_item == "cache_way_size") {
            model_config.cache_way_size = value;
            continue;
        }

        if (str_item == "cache_set_size") {
            model_config.cache_set_size = value;
            continue;
        }


        if (str_item == "allocate_on_miss") {
            model_config.allocate_on_miss = value;
            continue;
        }

        if (str_item == "jam_instruction") {
            model_config.jam_instruction = value;
            continue;
        }

        if (str_item == "latency_type") {
            model_config.latency_type = value;
            continue;
        }

        if (str_item == "latency_mean") {
            model_config.latency_mean = value;
            continue;
        }

        if (str_item == "latency_dev") {
            model_config.latency_dev = value;
            continue;
        }

        std::cout << "#### read_model_config_from_file: Unrecognized config item: " << line << std::endl;
        continue;
    }

    model_config.print();
}


int read_trace_from_file(std::string file_path, std::vector<WarpTrace> &warp_traces, ThreadDim &thread_dim) {
    std::ifstream in_stream;

    //  Open the trace file
    //  If the file can not be opened, return error
    in_stream.open(file_path, std::ifstream::in);
    if (! in_stream.is_open()) {
        std::cout << "####  read_trace_from_file: trace file '" << file_path << "' can not be opened  ####" << std::endl;
        return -1;
    }

    //  Read thread dimension information from trace file
    int block_size, grid_size;
    in_stream >> block_size >> grid_size;
    thread_dim.reset(block_size, grid_size);

    //  Allocate space for warp_accesses
    warp_traces.reserve(thread_dim.num_warps);
    warp_traces.resize(thread_dim.num_warps);

    //  Read traces from the trace file
    int warp_id;
    int pc;
    int jam;
    int width;
    int num_valid_accesses;
    unsigned long long *addr = new unsigned long long[thread_dim.threads_per_warp];

    in_stream >> warp_id >> pc >> width >> jam >> num_valid_accesses;
    while (in_stream.good()) {
        int i;

        for (i = 0; i < num_valid_accesses; i++)
            in_stream >> addr[i];

        warp_traces[warp_id].add_warp_access(pc, width, jam, num_valid_accesses, addr);
    }
    delete[] addr;


    return 0;
}
