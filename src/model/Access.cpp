


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


GlobalAccess::GlobalAccess() {
}

WarpAccess::WarpAccess() {
    accesses = NULL;
    size = 0;
    jam = 0;
    pc = 0;
}

WarpAccess::WarpAccess(int p, int w, int j, int s, unsigned long long *addr) {
    int i;

    this->pc = p;
    this->jam = j;
    this->size = s;

    accesses = new GlobalAccess[s];
    for (i = 0; i < s; i++) {
        accesses[i].width = w;
        accesses[i].address = addr[i];
    }
}

WarpAccess::~WarpAccess() {
    if (size > 0)
        delete[] accesses;
}

int WarpAccess::get_num_distinct_block_addr() {


    return 0;
}

void WarpAccess::coalesce(ModelConfig & model_config, ThreadDim & thread_dim) {
    int coalesce_width;

    //  Calculate the coalesce width
    coalesce_width = thread_dim.threads_per_warp;
    if (accesses[0].width == 8)
        coalesce_width = thread_dim.threads_per_warp / 2;
    if (accesses[0].width == 16)
        coalesce_width = thread_dim.threads_per_warp / 4;
}

WarpTrace::WarpTrace() {
}

void WarpTrace::add_warp_access(int p,int w, int j, int s, unsigned long long *addr) {
    warp_accesses.emplace_back(p, w, j, s, addr);
}

void WarpTrace::coalesce(ModelConfig & model_config, ThreadDim & thread_dim) {
    std::vector<WarpAccess>::iterator it;
    for (it = warp_accesses.begin(); it != warp_accesses.end(); it++) {
        it->coalesce(model_config);
    }
}
