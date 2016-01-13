


#include "MyLastLoad.h"


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
            for (j = i; j < i + WARP_SIZE && j < block_size; j++) {
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
    num_warps_per_block = (b_size - 1) / WARP_SIZE + 1;
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

