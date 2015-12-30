

#include <iostream>
#include <fstream>
#include <sstream>
#include "ModelConfig.h"
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

    in_stream.open(file_path, std::ofstream::in);
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
