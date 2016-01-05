


#include <iostream>
#include <vector>
#include <set>
#include <string>

#include "constants.h"
#include "ModelConfig.h"

#ifndef MY_WARP_ACCESS
#define MY_WARP_ACCESS

class ThreadDim{
	public:
	int threads_per_warp;
	int warps_per_block;
	int num_blocks;
	int num_warps;
	int num_threads;

	public:
	ThreadDim();
	void reset(int block_size, int grid_size);
};

typedef unsigned long long addr_type;

class WarpAccess {
	public:
	addr_type *accesses;
	int size;
	int jam;
	int pc;
    int width;

	public:
	WarpAccess();
	WarpAccess(int _pc, int _width, int _jam, int _size, addr_type *addr);
	~WarpAccess();
};

class WarpTrace {
	private:
	std::vector<WarpAccess> warp_accesses;
    int location;

	public:
	WarpTrace();
	void add_warp_access(int _pc,int _width, int _jam, int _size, addr_type *addr);

    void reset();
    bool is_finish();
    WarpAccess * next_warp_access();
};


#endif
