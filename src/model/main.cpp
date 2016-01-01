


#include <iostream>
#include <vector>

#include "functions.h"
#include "ModelConfig.h"
#include "Access.h"
#include "io.h"


int main(int argc, char **argv) {
    ModelConfig model_config;
    ThreadDim thread_dim;
    std::vector<WarpTrace> warp_traces;


    //  Number of arguments check
    //  Argument 0: executable file name
    //  Argument 1: input trace file path
    //  Argument 2: output file path
    //  Argument 3: config file path
    if (argc != 4) {
        std::cout << "####  main: Too many or too few arguments.  ####" << std::endl;
        return -1;
    }

    //  Read model config from coresponding file
    std::cout << "####  main: Reading model config from '" << argv[3] << "'  ####" << std::endl;
    read_model_config_from_file(argv[3], model_config);

    //  Read input trace from file
    std::cout << "####  main: Reading trace from '" << argv[1] << "'  ####" << std::endl;
    read_trace_from_file(argv[1], warp_traces, thread_dim);

    //  Do the coalescing




    return 0;
}
