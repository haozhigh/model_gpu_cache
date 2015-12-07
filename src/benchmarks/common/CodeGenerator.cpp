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
	std::string code_out_path;
    std::string kernel_name;
    std::vector<std::string> instructions;

	std::string demangle(std::string str_in);
	std::string strip_parameters(std::string full_name);
	void force_exit();
    void write_to_file();

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
    std::cout<< "##  Force Exit  ##" << std::endl;
	std::_Exit(-1);
}

void TraceGenerator::write_to_file() {
	std::ofstream out_stream;

	//  Open the file to write traces
	out_stream.open(code_out_path + "/" + kernel_name + ".ptx", std::ofstream::out);
	if (! out_stream.is_open()) {
		std::cout << "####  Failed to open file to write code  ####" <<std::endl;
        std::cout << "####  Program exiting ####" << std::endl;
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

TraceGenerator::TraceGenerator(std::string _code_out_path) {
	code_out_path = _code_out_path;
    kernel_name = "";
}

void TraceGenerator::initialize(const executive::ExecutableKernel & kernel) {
    //  If it is not the first kernel of this program, write instructions to file
    if (kernel_name != "") {
        write_to_file();
    }

    //  Reset instructions to empty vector
    instructions.clear();
    std::vector<std::string> ().swap(instructions);

    //  Get name of the kernel, and print it to console
	kernel_name = this->strip_parameters(this->demangle(kernel.name));
    std::cout<< "####  Start Generating code for " << kernel_name << "  ####" << std::endl;

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
}

void TraceGenerator::event(const trace::TraceEvent & event) {
    int pc;

    pc = event.instruction->pc;
    if (pc + 1 > instructions.size()) {
        instructions.resize(pc + 1);
    }

    if (instructions[pc] == "") {
        instructions[pc] = event.instruction->toString();
    }
}

void TraceGenerator::finish() {
	//  Write instructions to file
    if (kernel_name != "")
        write_to_file();
}


extern int original_main(int, char**);

int main(int argc, char** argv) {
	TraceGenerator generator(argv[1]);
	ocelot::addTraceGenerator(generator);
	original_main(argc - 1, argv + 1);
}
