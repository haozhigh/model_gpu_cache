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


class TraceGenerator : public trace::TraceGenerator {
    private:
        std::string trace_out_path;
        std::string kernel_name;

        //  Record ptx instructions of each kernel
        std::vector<std::string> instructions;

        //  Record compute and memory instructions executed
        long long compute_count;
        long long memory_count;

        std::string demangle(std::string str_in);
        std::string strip_parameters(std::string full_name);
        void force_exit();

        //  Write ptx instructions to file
        void write_instructions();

        //  Write compute and memory count to file
        void write_instruction_count();

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

    //  Init instructions to record instructions for this kernel
    instructions.clear();
    std::vector<std::string> ().swap(instructions);

    //  Init compute and memory counter
    compute_count = 0;
    memory_count = 0;
}

void TraceGenerator::event(const trace::TraceEvent & event) {
    int block_id;
    int block_dim;

    //  Compute block_id and block_dim
    block_id = event.blockId.x * event.gridDim.y * event.gridDim.z + event.blockId.y * event.gridDim.z + event.blockId.z;
    block_dim = event.blockDim.x * event.blockDim.y * event.blockDim.z;

    //  Record instructions
    int pc = event.instruction->pc;
    if (pc + 1 > instructions.size()) {
        instructions.resize(pc + 1);
    }
    if (instructions[pc] == "") {
        instructions[pc] = event.instruction->toString();
    }

    //  Count compute and memory instructions executed
    if (event.instruction->addressSpace == ir::PTXInstruction::Global) {
        //  If it is a global memory instruction
        for (int i = 0; i < block_dim; i++)
            if (event.active[i])
                memory_count ++;
    }
    else {
        //  If it is not a global memory instruction
        for (int i = 0; i < block_dim; i++)
            if (event.active[i])
                compute_count ++;
    }
}

//  Should be called whenever a kernel finishes
void TraceGenerator::finish() {
    //  Write instructions to file for this kernel
    this->write_instructions();

    //  Write instruction count to file for this kernel
    this->write_instruction_count();
}

void TraceGenerator::write_instructions() {
	std::ofstream out_stream;

	//  Open the file to write traces
	out_stream.open(trace_out_path + "/" + kernel_name + ".ptx", std::ofstream::out);
	if (! out_stream.is_open()) {
		std::cout << "####  TraceGenerator::write_instructions: Failed to open file to write code  ####" <<std::endl;
		this->force_exit();
	}

    //  Write instructions to file
    int i;
    for (i = 0; i < instructions.size(); i++) {
        out_stream << std::setw(5) << std::left << (i + 1) << instructions[i] << "\n";
    }

    //  Close file
    out_stream.close();
}

void TraceGenerator::write_instruction_count() {
    std::ofstream out_stream;

    //  Open the file
    out_stream.open(trace_out_path + "/" + kernel_name + ".count", std::ofstream::out);
    if (! out_stream.is_open()) {
        std::cout << "#### TraceGenerator::write_instruction_count: Failed to open file to write ####" << std::endl;
        this->force_exit();
    }

    //  Write instruction count to file
    out_stream << compute_count << " " << memory_count;

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
