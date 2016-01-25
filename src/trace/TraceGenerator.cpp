//  Ocelot includes
#include <ocelot/api/interface/ocelot.h>
#include <ocelot/trace/interface/TraceGenerator.h>
#include <ocelot/trace/interface/TraceEvent.h>
#include <ocelot/executive/interface/ExecutableKernel.h>
#include <ocelot/ir/interface/PTXInstruction.h>
#include <ocelot/ir/interface/PTXOperand.h>

//  C++ includes
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <string>


#include "MyLastLoad.h"


#define TOTAL_NUM_ACCESS_LIMIT 10000000


class TraceGenerator : public trace::TraceGenerator {
    private:
        std::string trace_out_path;
        std::ofstream out_stream;
        std::string kernel_name;

        std::string demangle(std::string str_in);
        std::string strip_parameters(std::string full_name);
        void force_exit();

        //  Variables needed to limit the total number of accesses recorded
        bool num_access_within_limit;
        int total_num_accesses;
        int last_block_id;

        //  Variable to achieve jam info recording
        MyLastLoad last_load;

        //  Record block dimension infomation
        int total_threads_per_block;
        int total_threads;
        void write_total_num_threads_to_file();

    public:
        TraceGenerator(std::string _trace_out_path);
        void initialize(const executive::ExecutableKernel & kernel);
        void event(const trace::TraceEvent & event);
        void finish();

};

std::string TraceGenerator::demangle(std::string str_in) {
    std::string str_out;
    FILE *pipe;
    std::string command;
    char buffer[128];

    command = "c++filt \"" + str_in + "\"";
    pipe = popen(command.c_str(), "r");
    if (pipe == NULL) {
        return "ERROR";
    }

    while (fgets(buffer, 128, pipe) != NULL) {
        str_out += buffer;
    }
    pclose(pipe);

    return str_out;
}

std::string TraceGenerator::strip_parameters(std::string full_name) {
    int locate;

    locate = full_name.find("(");
    if (locate == -1)
        return full_name;
    else
        return full_name.substr(0, locate);
}

void TraceGenerator::force_exit() {
    //  Force exit program executing
    std::cout << "#### TraceGenerator::force_exit: Force Exit ####" << std::endl;
    std::_Exit(-1);
}

TraceGenerator::TraceGenerator(std::string _trace_out_path) {
	this->trace_out_path = _trace_out_path;
}

void TraceGenerator::initialize(const executive::ExecutableKernel & kernel) {
    //  Get name of the kernel, and print it to console
	kernel_name = this->strip_parameters(this->demangle(kernel.name));
    std::cout<< "####  TraceGenerator::initialize: Start Generating trace for " << kernel_name << "  ####" << std::endl;

	//  Check if the kernel is among the most time-consuming ones.
	//  If so, exit the program
	std::vector<std::string> avoid_kernels = {	"corr_kernel",
												"covar_kernel",
												"BFS_kernel_multi_blk_inGPU"
												};
	std::vector<std::string>::iterator it;
	for (it = avoid_kernels.begin(); it != avoid_kernels.end(); ++it) {
		if (*it == kernel_name) {
			std::cout << "####  TraceGenerator::initialize: " << kernel_name << " encountered  ####" << std::endl;
            std::cout << "####  TraceGenerator::initialize: Program exiting ####" << std::endl;
			this->force_exit();
		}
	}

	//  Check if this kernel has been executed already
	//  Exit program while seeing repeated kernels
	static std::vector<std::string> executed_kernels;
	for (it = executed_kernels.begin(); it != executed_kernels.end(); ++it) {
		if (*it == kernel_name) {
			std::cout << "####  TraceGenerator::initialize: Repeated " << kernel_name << " encountered  ####" << std::endl;
            std::cout << "####  TraceGenerator::initialize: Program exiting ####" << std::endl;
			this->force_exit();
		}
	}
	executed_kernels.push_back(kernel_name);

	//  Open the file to write traces
	this->out_stream.open(this->trace_out_path + "/" + kernel_name + ".trc", std::ofstream::out);
	if (! this->out_stream.is_open()) {
		std::cout << "####  TraceGenerator::initialize: Failed to open file to write trace  ####" <<std::endl;
        std::cout << "####  TraceGenerator::initialize: Program exiting ####" << std::endl;
		this->force_exit();
	}

    //  Write block dimension info to trace file
    //  Set total number of threads to grid dimension for now
    ir::Dim3 block_dim;
    ir::Dim3 grid_dim;
    block_dim = kernel.blockDim();
    grid_dim = kernel.gridDim();
    total_threads_per_block = block_dim.x * block_dim.y * block_dim.z;
    total_threads = grid_dim.x * grid_dim.y * grid_dim.z * total_threads_per_block;
    //this->out_stream << block_dim.x << " ";
    //this->out_stream << block_dim.y << " ";
    //this->out_stream << block_dim.z << "\n";

    //  Reset variables to limit total number of accesses
    num_access_within_limit = true;;
    total_num_accesses = 0;
    last_block_id = -1;

    //  Assign memory space for last_load
    int block_size;
    block_size = block_dim.x * block_dim.y * block_dim.z;
    last_load.assign_memory(block_size);
}

void TraceGenerator::event(const trace::TraceEvent & event) {
    int block_id;
    int block_dim;

    //  If the number of accesses already exceeds limit, do not record any more
    if (! num_access_within_limit)
        return;

    //  Compute block_id and block_dim
    block_id = event.blockId.x * event.gridDim.y * event.gridDim.z + event.blockId.y * event.gridDim.z + event.blockId.z;
    block_dim = event.blockDim.x * event.blockDim.y * event.blockDim.z;

    // At the start of each thread block
    if (block_id > last_block_id) {
        last_block_id = block_id;

        //  If last_load of the last thread block is not empty, write it to file
        last_load.write_to_file(this->out_stream);

        // Check if the number of accesses exceeds limit
        if (total_num_accesses > TOTAL_NUM_ACCESS_LIMIT) {
            num_access_within_limit = false;
            std::cout<< "####  TraceGenerator::event: Total number of accesses exceeds " << TOTAL_NUM_ACCESS_LIMIT << "  ####" << std::endl;
            int grid_dim = event.gridDim.x * event.gridDim.y * event.gridDim.z;
            std::cout<< "####  TraceGenerator::event: Please wait while " << (grid_dim - block_id) << "(out of " << grid_dim << ") blocks are still running ####" << std::endl;

            //  Reset total number of theads to the restricted value
            total_threads = block_dim * block_id;

            //  Return for the first event that number of accesses exceeds limit
            return;
        }
    }
    
    //  Only handles global load instructions or Tex instructions
    if ((event.instruction->isLoad() && event.instruction->addressSpace == ir::PTXInstruction::Global) ||
        event.instruction->opcode == ir::PTXInstruction::Tex) {

        //  If last_load is not empty, write it to file
        //  Only write warps that are going to be affected by current event
        last_load.write_to_file(this->out_stream, event);

        //  Update last_load with current event
        last_load.update(event);

        //  Increase number of accesses by thread block dimension
        total_num_accesses += block_dim;
    }
    //  If it is not a global load event
    else {
        //  If last_load is not empty, check jam info
        last_load.check_jam(event);
    }
}

//  Should be called whenever a kernel finishes
void TraceGenerator::finish() {
    //  If last_load of the last thread block is not empty, write it to file
    //  Release memory for last_load
    last_load.write_to_file(this->out_stream);
    last_load.release_memory();

	//  If out_stream is opened for last kernel, close it
	if (this->out_stream.is_open())
		out_stream.close();

    //  Write total number of threads to file
    this->write_total_num_threads_to_file();
}

void TraceGenerator::write_total_num_threads_to_file() {
    std::ofstream out_stream;

    //  Open the file to write
    out_stream.open(trace_out_path + "/" + kernel_name + ".trc.dim", std::ofstream::out);
    if (! out_stream.is_open()) {
		std::cout << "####  TraceGenerator::write_total_num_threads_to_file: Failed to open file to write code  ####" <<std::endl;
		this->force_exit();
    }
    
    //  Do the write
    int total_blocks;
    total_blocks = total_threads / total_threads_per_block;
    out_stream << this->total_threads_per_block << " ";
    out_stream << total_blocks << "\n";

    //  Close file
    out_stream.close();
}

extern int original_main(int, char**);

int main(int argc, char** argv) {
	TraceGenerator generator(argv[1]);
	ocelot::addTraceGenerator(generator);
	original_main(argc - 1, argv + 1);
	return 0;
}
