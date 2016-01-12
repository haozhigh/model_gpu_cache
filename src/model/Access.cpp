


#include "Access.h"


ThreadDim::ThreadDim() {
    threads_per_warp = 32;

    warps_per_block = 0;
    num_threads = 0;
    num_warps = 0;
    num_blocks = 0;
}

void ThreadDim::reset(int block_size, int grid_size) {
    threads_per_warp = 32;
    warps_per_block = block_size / threads_per_warp;

    num_blocks = grid_size;
    num_warps = num_blocks * warps_per_block;
    num_threads = num_warps * threads_per_warp;
}


WarpAccess::WarpAccess() {
    accesses = NULL;
    size = 0;
    jam = 0;
    pc = 0;
    width = 0;
}

WarpAccess::WarpAccess(const WarpAccess & _warp_access) {
    int i;

    this->pc = _warp_access.pc;
    this->width = _warp_access.width;
    this->jam = _warp_access.jam;
    this->size = _warp_access.size;

    accesses = new addr_type[this->size];
    for (i = 0; i < this->size; i++) {
        accesses[i] = _warp_access.accesses[i];
    }
}

WarpAccess::WarpAccess(int _pc, int _width, int _jam, int _size, addr_type *addr) {
    int i;

    this->pc = _pc;
    this->width = _width;
    this->jam = _jam;
    this->size = _size;

    accesses = new addr_type[_size];
    for (i = 0; i < _size; i++) {
        accesses[i] = addr[i];
    }
}

WarpAccess::~WarpAccess() {
    if (size > 0)
        delete[] accesses;
}

WarpTrace::WarpTrace() {
    this->reset();
}

void WarpTrace::reset() {
    this->jam = 0;
    this->location = 0;
}

void WarpTrace::set_jam(int time_stamp) {
    this->jam = time_stamp;
}

bool WarpTrace::is_available(int time_stamp) {
    return ((! this->is_finish()) && time_stamp >= jam);
}

bool WarpTrace::is_finish() {
    if (location < warp_accesses.size())
        return false;
    else
        return true;
}

WarpAccess * WarpTrace::next_warp_access(int time_stamp) {
    if (this->is_finish() || time_stamp < jam)
        return NULL;
    else {
        location ++;
        return &warp_accesses[location - 1];
    }
}

void WarpTrace::add_warp_access(int _pc,int _width, int _jam, int _size, addr_type *addr) {
    warp_accesses.push_back(WarpAccess(_pc, _width, _jam, _size, addr));
}
