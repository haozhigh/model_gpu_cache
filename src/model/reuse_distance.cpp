



#include "reuse_distance.h"
#include "functions.h"
#include "tree.h"


#define STACK_EXTRA_SIZE 8

void calculate_reuse_distance(AnalyseTask & task, ModelConfig & model_config, DistanceStat & stat) {
    int i, j;
    int fake_stamp;                         //  Record number of memory access
    std::vector<int> total_accesses_per_set;
    WarpAccess * p_warp_access;
    int set_id;

    //  Calculate total number of accesses for each set
    task.reset();
    total_accesses_per_set.resize(model_config.cache_set_size, 0);
    p_warp_access = task.next_warp_access();
    while (p_warp_access != NULL) {
        for (i = 0; i < p_warp_access->size; i++) {
            set_id = calculate_cache_set(p_warp_access->accesses[i], model_config);
            total_accesses_per_set[set_id] ++;
        }

        p_warp_access = task.next_warp_access();
    }

	// Create a tree data structure for each set (B in the Almasi et al. paper)
	std::vector<Tree> B;
	B.reserve(model_config.cache_set_size);
	for (set_id = 0; set_id < model_config.cache_set_size; set_id ++) {
		B.emplace_back(total_accesses_per_set[set_id] + STACK_EXTRA_SIZE);
	}

	// Create the hash data structure (P in the Almasi et al. paper)
	std::map<addr_type, int> P;
	
	

}
