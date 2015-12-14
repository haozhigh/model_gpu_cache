


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


#define TOTAL_NUM_ACCESS_LIMIT 10000000

class MyMemoryAccess {
    public:
    int pc;
    unsigned long long address;
    int width;
    int jam;
    int target;

    MyMemoryAccess();
};

MyMemoryAccess::MyMemoryAccess() {
    pc = 0;
    address = 0;
    width = 0;
    jam = 0;
    target = 0;
}

class MyLastLoad {
    private:
    MyMemoryAccess *accesses;
    int block_size;
    int block_id;
    bool valid;

    public:
    MyLastLoad();
    bool is_valid();
    void update(const trace::TraceEvent &event);
    void check_jam(const trace::TraceEvent &event);
    void assign_memory(int b_size);
    void release_memory();
    void write_to_file(std::ofstream &out_stream);
    ~MyLastLoad();
};

MyLastLoad::MyLastLoad() {
    accesses = NULL;
    block_size = 0;
    block_id = 0;
    valid = false;
}

bool MyLastLoad::is_valid() {
    return this->valid;
}

void MyLastLoad::write_to_file(std::ofstream &out_stream) {
    if (! valid)
        return;

    int i;
    for (i = 0; i < block_size; i++) {
        int thread_id;

        thread_id = block_size * block_id + i;
        out_stream << thread_id << " ";
        out_stream << accesses[i].pc << " ";
        out_stream << accesses[i].address << " ";
        out_stream << accesses[i].width << " ";
        out_stream << accesses[i].jam << "\n";
    }

    valid = false;
}

void MyLastLoad::update(const trace::TraceEvent &event) {
    //  Update accesses with current event
    int i;
    int pc;
    int width;
    int address_counter;

    pc = event.instruction->pc;
    width = event.instruction->vec * ir::PTXOperand::bytes(event.instruction->type);
    address_counter = 0;
    for (i = 0; i < block_size; i++) {
            //  Check if this thread launches a memory access
            if (event.active[i]) {
                accesses[i].address = event.memory_addresses[address_counter];
                address_counter ++;
            }
            else {
                accesses[i].address = 0;
            }

            accesses[i].pc = pc;
            accesses[i].width = width;
            accesses[i].jam = 0;
    }

    //  Update block_id with this event
    block_id = event.blockId.x * event.gridDim.y * event.gridDim.z + event.blockId.y * event.gridDim.z + event.blockId.z;
}

void MyLastLoad::check_jam(const trace::TraceEvent &event) {

}

void MyLastLoad::assign_memory(int b_size) {
    release_memory();
    block_size = b_size;
    accesses = new MyMemoryAccess[block_size];
}

void MyLastLoad::release_memory() {
    if (accesses != NULL) {
        delete[] accesses;
        accesses = NULL;
    }
}

MyLastLoad::~MyLastLoad() {
    release_memory();
}

class TraceGenerator : public trace::TraceGenerator {
	private:
	std::string trace_out_path;
	std::ofstream out_stream;
    std::string kernel_name;
    bool kernel_started;
    std::vector<std::string> instructions;

	std::string demangle(std::string str_in);
	std::string strip_parameters(std::string full_name);
	void force_exit();
    void kernel_finish();
    void write_instructions();

    //  Variables needed to limit the total number of accesses recorded
    bool num_access_within_limit;
    int total_num_accesses;
    int last_block_id;

    //  Variable to achieve jam info recording
    MyLastLoad last_load;

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
    std::cout << "#### Force Exit ####" << std::endl;
	std::_Exit(-1);
}

TraceGenerator::TraceGenerator(std::string _trace_out_path) {
	this->trace_out_path = _trace_out_path;
    kernel_started = false;
}

void TraceGenerator::initialize(const executive::ExecutableKernel & kernel) {
    //  If kernel_name is empty, call kernel_finish for last kernel
    if (kernel_started) {
        this->kernel_finish();
    }

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

	//  Open the file to write traces
	this->out_stream.open(this->trace_out_path + "/" + kernel_name + ".trc", std::ofstream::out);
	if (! this->out_stream.is_open()) {
		std::cout<< "####  Failed to open file to write trace  ####" <<std::endl;
        std::cout << "####  Program exiting ####" << std::endl;
		this->force_exit();
	}

    //  Write block dimension info to trace file
    ir::Dim3 block_dim;
    block_dim = kernel.blockDim();
    this->out_stream << block_dim.x << " ";
    this->out_stream << block_dim.y << " ";
    this->out_stream << block_dim.z << "\n";
    /*std::cout << "helloworld1\n";
    std::cout << block_dim.x << " ";
    std::cout << block_dim.y << " ";
    std::cout << block_dim.z << "\n";
    std::cout << "helloworld2\n";
    this->out_stream.close();
    this->force_exit();*/

    //  Reset variables to limit total number of accesses
    num_access_within_limit = true;;
    total_num_accesses = 0;
    last_block_id = -1;

    //  Init instructions to record instructions for this kernel
    instructions.clear();
    std::vector<std::string> ().swap(instructions);

    //  Assign memory space for last_load
    int block_size;
    block_size = block_dim.x * block_dim.y * block_dim.z;
    last_load.assign_memory(block_size);

    //  Set kernel_started to true
    kernel_started = true;
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

        // Check if the number of accesses exceeds limit
        if (total_num_accesses > TOTAL_NUM_ACCESS_LIMIT) {
            num_access_within_limit = false;
            std::cout<< "####  Total number of accesses exceeds " << TOTAL_NUM_ACCESS_LIMIT << "  ####" << std::endl;
            int grid_dim = event.gridDim.x * event.gridDim.y * event.gridDim.z;
            std::cout<< "####  Please wait while " << (grid_dim - block_id) << "(out of " << grid_dim << ") blocks are still running ####" << std::endl;

            //  Return for the first event that number of accesses exceeds limit
            return;
        }
    }
    
    //  Record instructions
    int pc = event.instruction->pc;
    if (pc + 1 > instructions.size()) {
        instructions.resize(pc + 1);
    }
    if (instructions[pc] == "") {
        instructions[pc] = event.instruction->toString();
    }

    //  Only handles global load instructions or Tex instructions
    if ((event.instruction->isLoad() && event.instruction->addressSpace == ir::PTXInstruction::Global) ||
        event.instruction->opcode == ir::PTXInstruction::Tex) {

        //  If last_load is not empty, write it to file
        if (last_load.is_valid())
            last_load.write_to_file(this->out_stream);

        //  Update last_load with current event
        last_load.update(event);

        //  Increase number of accesses by thread block dimension
        total_num_accesses += block_dim;
    }
    //  If it is not a global load event
    else {
    }
}

//  Should be called whenever a kernel finishes
void TraceGenerator::kernel_finish() {
	//  If out_stream is opened for last kernel, close it
	if (this->out_stream.is_open())
		out_stream.close();

    //  Write instructions to file for this kernel
    this->write_instructions();

    kernel_started = false;
}

void TraceGenerator::finish() {
    //  Call kernel_finish for the last non-empty kernel executed
    if (kernel_started) {
        this->kernel_finish();
    }
}

void TraceGenerator::write_instructions() {
	std::ofstream out_stream;

	//  Open the file to write traces
	out_stream.open(trace_out_path + "/" + kernel_name + ".ptx", std::ofstream::out);
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

extern int original_main(int, char**);

int main(int argc, char** argv) {
	TraceGenerator generator(argv[1]);
	ocelot::addTraceGenerator(generator);
	original_main(argc - 1, argv + 1);
	return 0;
}
