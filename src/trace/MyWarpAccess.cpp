


#include "MyWarpAccess.h"

MyAccess::MyAccess() {
    address = 0;
}


MyWarpAccess::MyWarpAccess() {
    this->reset();
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

