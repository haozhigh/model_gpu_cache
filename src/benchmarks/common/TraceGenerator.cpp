


//  Ocelot includes
#include <ocelot/api/interface/ocelot.h>
#include <ocelot/trace/interface/TraceGenerator.h>
#include <ocelot/trace/interface/TraceEvent.h>
#include <ocelot/executive/interface/ExecutableKernel.h>
#include <ocelot/ir/interface/PTXInstruction.h>
#include <ocelot/ir/interface/PTXOperand.h>

//  C++ includes
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>


#define TOTAL_NUM_ACCESS_LIMIT 10000000

class TraceGenerator : public trace::TraceGenerator {
	private:
	std::string trace_out_path;
	std::ofstream out_stream;

	std::string demangle(std::string str_in);
	std::string strip_parameters(std::string full_name);
	void force_exit();

    //  Variables needed to limit the total number of accesses recorded
    bool num_access_within_limit;
    int total_num_accesses;
    int last_block_id;

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
	//  If out_stream is opened for last kernel, close it
	if (this->out_stream.is_open())
		out_stream.close();
	
	//  Force exit program executing
	std::_Exit(-1);
}

TraceGenerator::TraceGenerator(std::string _trace_out_path) {
	this->trace_out_path = _trace_out_path;
}

void TraceGenerator::initialize(const executive::ExecutableKernel & kernel) {
	std::string kernel_name;

    //  Get name of the kernel, and print it to console
	kernel_name = this->strip_parameters(this->demangle(kernel.name));
    std::cout<< "####  Start Generating trace for " << kernel_name << "  ####" << std::endl;

	//  Check if the kernel is among the most time-consuming ones.
	//  If so, exit the program
	std::vector<std::string> avoid_kernels = {	"corr_kernel",
												"covar_kernel",
												"BFS_kernel_multi_blk_inGPU"
												};
	std::vector<std::string>::iterator it;
	for (it = avoid_kernels.begin(); it != avoid_kernels.end(); ++it) {
		if (*it == kernel_name) {
			std::cout << "####  " << kernel_name << " encountered  ####" << std::endl;
            std::cout << "####  Program exiting ####" << std::endl;
			this->force_exit();
		}
	}

	//  Check if this kernel has been executed already
	//  Exit program while seeing repeated kernels
	static std::vector<std::string> executed_kernels;
	for (it = executed_kernels.begin(); it != executed_kernels.end(); ++it) {
		if (*it == kernel_name) {
			std::cout << "####  Repeated " << kernel_name << " encountered  ####" << std::endl;
            std::cout << "####  Program exiting ####" << std::endl;
			this->force_exit();
		}
	}
	executed_kernels.push_back(kernel_name);

	//  If out_stream is opened for last kernel, close it
	if (this->out_stream.is_open())
		out_stream.close();

	//  Open the file to write traces
	this->out_stream.open(this->trace_out_path + "/" + kernel_name + ".trc", std::ofstream::out);
	if (! this->out_stream.is_open()) {
		std::cout<< "####  Failed to open file to write trace  ####" <<std::endl;
        std::cout << "####  Program exiting ####" << std::endl;
		this->force_exit();
	}

    //  Reset variables to limit total number of accesses
    num_access_within_limit = true;;
    total_num_accesses = 0;
    last_block_id = -1;

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

    //  At the first event of each kernel, write block dimension info to trace file
    if (last_block_id == -1) {
        this->out_stream << event.blockDim.x << " ";
        this->out_stream << event.blockDim.y << " ";
        this->out_stream << event.blockDim.z << "\n";
    }

    // At the first event of each thread block, check if the number of accesses exceeds limit
    if (block_id > last_block_id) {
        last_block_id = block_id;
        if (total_num_accesses > TOTAL_NUM_ACCESS_LIMIT) {
            num_access_within_limit = false;
            std::cout<< "####  Total number of accesses exceeds " << TOTAL_NUM_ACCESS_LIMIT << "  ####" << std::endl;
            int grid_dim = event.gridDim.x * event.gridDim.y * event.gridDim.z;
            std::cout<< "####  Please wait while " << (grid_dim - block_id) << "(out of " << grid_dim << ") blocks are still running ####" << std::endl;
        }
    }

    //  Only handles global load instructions or Tex instructions
    if ((event.instruction->isLoad() && event.instruction->addressSpace == ir::PTXInstruction::Global) ||
        event.instruction->opcode == ir::PTXInstruction::Tex) {
        //  Infomation that needs to save
        unsigned int thread_id;
        unsigned int access_size;
        unsigned long long access_address;
        unsigned int program_counter;

        //  Loop over each thread in this thread block
        unsigned memory_address_counter = 0;
        program_counter = event.instruction->pc;
        for (unsigned int local_thread_id = 0; local_thread_id < block_dim; local_thread_id++) {
            thread_id = block_id * block_dim + local_thread_id;

            //  Check if this thread launches a memory access
            if (event.active[local_thread_id]) {
                access_address = event.memory_addresses[memory_address_counter];
                access_size = event.instruction->vec * ir::PTXOperand::bytes(event.instruction->type);
            }
            else {
                access_address = 0;
                access_size = 0;
            }

            //  Output the access info to trace file
            this->out_stream << thread_id << " ";
            this->out_stream << program_counter << " ";
            this->out_stream << access_address << " ";
            this->out_stream << access_size << "\n";
        }

        //  Increase number of accesses by thread block dimension
        total_num_accesses += block_dim;
    }
}

void TraceGenerator::finish() {
	//  If out_stream is opened for last kernel, close it
	if (this->out_stream.is_open())
		out_stream.close();

}


extern int original_main(int, char**);

int main(int argc, char** argv) {
	TraceGenerator generator(argv[1]);
	ocelot::addTraceGenerator(generator);
	original_main(argc - 1, argv + 1);
	return 0;
}
