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
#define WARP_SIZE 32

class MyAccess {
    public:
    unsigned long long address;

    MyAccess();
};

MyAccess::MyAccess() {
    address = 0;
}

class MyWarpAccess {
    private:
    MyAccess accesses[WARP_SIZE];
    int global_warp_id;
    int pc;
    int target;
    int width;
    int jam;
    int valid;
    int num_valid_accesses;

    public:
    void write_to_file(std::ofstream &out_stream);
    void reset();
    void set_warp_info(int gwi, int p, int t, int width);
    void add(unsigned long long a);
    bool is_valid();
    bool is_jam();
    void check_jam(int reg_id);
    

    MyWarpAccess();
};

MyWarpAccess::MyWarpAccess() {
    valid = 0;
    num_valid_accesses = 0;
}

bool MyWarpAccess::is_valid() {
    return (valid == 1);
}

bool MyWarpAccess::is_jam() {
    return (jam == 1);
}

void MyWarpAccess::reset() {
    valid = 0;
    num_valid_accesses = 0;
}

void MyWarpAccess::set_warp_info(int gwi, int p, int t, int w) {
    global_warp_id = gwi;
    pc = p;
    target = t;
    width = w;
    jam = 0;
}

void MyWarpAccess::add(unsigned long long a) {
    this->valid = 1;
    accesses[num_valid_accesses].address = a;
    num_valid_accesses ++;
}

void MyWarpAccess::check_jam(int reg_id) {
    if (this->is_jam())
        return;

    if (reg_id >= target && (reg_id < target + width))
        jam = 1;
    return;
}

void MyWarpAccess::write_to_file(std::ofstream &out_stream) {
    //  Check and reset valid flag of this warp access
    if (valid != 1)
        return;

    //  Output global warp id, pc valie, width, jam info, and number of valid accesses in this warp
    out_stream << global_warp_id << " ";
    out_stream << pc << " ";
    out_stream << width << " ";
    out_stream << jam << " ";
    out_stream << num_valid_accesses << " ";

    /*
    //  Calculate number of accesses in this warp, and output it
    int num_accesses_this_warp;
    num_accesses_this_warp = 0;
    for (j = 0; j < WARP_SIZE; j++)
        if (warp_accesses[i].accesses[j].valid == 1)
            num_accesses_this_warp ++;
    out_stream << num_accesses_this_warp << " ";
    */

    //  Output each single access address and width of this warp
    for (int i = 0; i < num_valid_accesses; i++) {
        //  Output access address and width
        out_stream << accesses[i].address << " ";
    }

    //  Outpub end of line
    out_stream << "\n";

    //  Reset itself
    this->reset();
}

class MyLastLoad {
    private:
        MyWarpAccess *warp_accesses;
        int block_size;
        int num_warps_per_block;
        int block_id;

        int strip_reg_number(const std::string str);

    public:
        MyLastLoad();
        void update(const trace::TraceEvent &event);
        void check_jam(const trace::TraceEvent &event);
        void assign_memory(int b_size);
        void release_memory();
        void write_to_file(std::ofstream &out_stream);
        void write_to_file(std::ofstream &out_stream, const trace::TraceEvent &event);
        ~MyLastLoad();
};

MyLastLoad::MyLastLoad() {
    warp_accesses = NULL;
    num_warps_per_block = 0;
    block_id = 0;
}

int MyLastLoad::strip_reg_number(const std::string str) {
    std::string tmp_str;

    //  If str is not a valid reg string, return -1
    if (str.size() < 3)
        return -1;

    tmp_str = str.substr(2);
    return atoi(tmp_str.c_str());
}

void MyLastLoad::write_to_file(std::ofstream &out_stream) {
    if (warp_accesses == NULL)
        return;

    int i;
    for (i = 0; i < num_warps_per_block; i++) {
        if (warp_accesses[i].is_valid())
            warp_accesses[i].write_to_file(out_stream);
    }
}

void MyLastLoad::write_to_file(std::ofstream &out_stream, const trace::TraceEvent &event) {
    if (warp_accesses == NULL)
        return;

    for (int i = 0; i < block_size; i++) {
        if (event.active[i]) {
            int local_warp_id;

            local_warp_id = i / WARP_SIZE;
            if (warp_accesses[local_warp_id].is_valid())
                warp_accesses[local_warp_id].write_to_file(out_stream);
        }
    }
}

void MyLastLoad::update(const trace::TraceEvent &event) {
    //  Update accesses with current event
    int i;
    int pc;
    int width;
    int target;
    int address_counter;

    //  Update block_id with this event
    block_id = event.blockId.x * event.gridDim.y * event.gridDim.z + event.blockId.y * event.gridDim.z + event.blockId.z;

    //  Get pc, width, and target
    pc = event.instruction->pc;
    width = event.instruction->vec * ir::PTXOperand::bytes(event.instruction->type);
    target = this->strip_reg_number(event.instruction->d.toString());

    //  Loop over all the threads in the block
    address_counter = 0;
    bool warp_active = false;
    for (i = 0; i < block_size; i++) {
        //  At the start of each warp, check if this warp has valid accesses
        if (i % WARP_SIZE == 0) {
            warp_active = false;

            int j;
            for (j = i; j < i + WARP_SIZE; j++) {
                if (event.active[j]) {
                    warp_active = true;
                    break;
                }
            }
        }

        //  If the current warp is valid, then record access of each thread
        //  Unactive access is marked by a zero address
        if (warp_active) {
            int address;
            int local_warp_id;

            //  Set warp info
            local_warp_id = i / WARP_SIZE;
            if (! warp_accesses[local_warp_id].is_valid()) {
                int global_warp_id;

                global_warp_id = block_id * num_warps_per_block + local_warp_id;
                warp_accesses[local_warp_id].set_warp_info(global_warp_id, pc, target, width);
            }

            //  Add address and width to the corresponding warp
            if (event.active[i]) {
                address = event.memory_addresses[address_counter];
                address_counter ++;
            }
            else {
                address = 0;
            }
            warp_accesses[local_warp_id].add(address);

        }
    }
}

void MyLastLoad::check_jam(const trace::TraceEvent &event) {
    int a, b, c;
    int local_warp_id;

    for (int i = 0; i < block_size;) {
        //  Only check jam for valid threads
        if (! event.active[i]) {
            i ++;
            continue;
        }

        //  Check jam
        local_warp_id = i / WARP_SIZE;
        a = this->strip_reg_number(event.instruction->a.toString());
        b = this->strip_reg_number(event.instruction->a.toString());
        c = this->strip_reg_number(event.instruction->a.toString());
        warp_accesses[local_warp_id].check_jam(a);
        warp_accesses[local_warp_id].check_jam(b);
        warp_accesses[local_warp_id].check_jam(c);

        //  Increase i to the next warp
        i = (i / WARP_SIZE + 1) * WARP_SIZE;
    }
}

void MyLastLoad::assign_memory(int b_size) {
    release_memory();
    block_size = b_size;
    num_warps_per_block = b_size / WARP_SIZE;
    warp_accesses = new MyWarpAccess[num_warps_per_block];
}

void MyLastLoad::release_memory() {
    if (warp_accesses != NULL) {
        delete[] warp_accesses;
        warp_accesses = NULL;
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
        std::vector<std::string> instructions;

        std::string demangle(std::string str_in);
        std::string strip_parameters(std::string full_name);
        void force_exit();
        void write_instructions();

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

    //  Init instructions to record instructions for this kernel
    instructions.clear();
    std::vector<std::string> ().swap(instructions);

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

    //  Write instructions to file for this kernel
    this->write_instructions();
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
    out_stream << this->total_threads_per_block << " ";
    out_stream << this->total_threads << "\n";

    //  Close file
    out_stream.close();
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

extern int original_main(int, char**);

int main(int argc, char** argv) {
	TraceGenerator generator(argv[1]);
	ocelot::addTraceGenerator(generator);
	original_main(argc - 1, argv + 1);
	return 0;
}
