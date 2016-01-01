


#include <iostream>
#include <vector>
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


class GlobalAccess {
    public:
    unsigned long long address;
    int width;

    GlobalAccess();
};

class WarpAccess {
	private:
	GlobalAccess *accesses;
	int size;
	int jam;
	int pc;

	int get_num_distinct_block_addr();

	public:
	WarpAccess();
	WarpAccess(int p, int w, int j, int s, unsigned long long *addr);
	~WarpAccess();

	void coalesce(ModelConfig & model_config, ThreadDim & thread_dim);
	
};

class WarpTrace {
	private:
	std::vector<WarpAccess> warp_accesses;

	public:
	WarpTrace();
	void add_warp_access(int p,int w, int j, int s, unsigned long long *addr);
	void coalesce(ModelConfig & model_config, ThreadDim & thread_dim);
};


#endif
