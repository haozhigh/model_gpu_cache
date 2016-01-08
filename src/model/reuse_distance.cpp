



#include "reuse_distance.h"
#include "functions.h"
#include "tree.h"
#include "NormalGenerator.h"


#define STACK_EXTRA_SIZE 8

void calculate_reuse_distance(AnalyseTask & task, ModelConfig & model_config, DistanceStat & stat) {
    int i;
    int fake_stamp;                         //  Record number of memory access
    WarpAccess * p_warp_access;
    int set_id;

    //  The normal latency genenrator
    NormalGenerator normal_generator(model_config.latency_mean, model_config.latency_dev);

    //  Variable to record on going requests
    //  Meant for mshr check
    std::multimap<int, int> ongoing_requests;

    //  Calculate total number of accesses for each set
    std::vector<int> total_accesses_per_set;
    task.reset();
    total_accesses_per_set.resize(model_config.cache_set_size, 0);
    p_warp_access = task.next_warp_access(0);
    while (p_warp_access != NULL) {
        for (i = 0; i < p_warp_access->size; i++) {
            set_id = calculate_cache_set(p_warp_access->accesses[i], model_config);
            total_accesses_per_set[set_id] ++;
        }

        p_warp_access = task.next_warp_access(0);
    }

	// Create a tree data structure for each set (B in the Almasi et al. paper)
	std::vector<Tree> Bs;
	Bs.reserve(model_config.cache_set_size);
	for (set_id = 0; set_id < model_config.cache_set_size; set_id ++) {
		Bs.emplace_back(total_accesses_per_set[set_id] + STACK_EXTRA_SIZE);
	}

	// Create the hash data structure (P in the Almasi et al. paper)
    std::vector<std::map<addr_type, int>> Ps;
    Ps.reserve(model_config.cache_set_size);
	
    //  Record processed number of accesses per cache set
    std::vector<int> set_counters;
    set_counters.resize(model_config.cache_set_size);
    for (i = 0; i < set_counters.size(); i++)
        set_counters[i] = 0;

	
    //  Start the main loop
    fake_stamp = 0;
    task.reset();
    while (! task.is_finish()) {
        //  Take a warptrace
        //  If all warps are jamed, get a NULL pointer, and thuns increase fake_stamp and continue
        p_warp_access = task.next_warp_access(fake_stamp);
        if (p_warp_access == NULL) {
            fake_stamp ++;
            continue;
        }

        //  Process ongoing requests only before mshr check 
        process_ongoing_requests(ongoing_requests, fake_stamp);

        //  If MSHR check fails
        //  Increast fake_stamp to a point where more mshrs will be available
        while (p_warp_access->size + ongoing_requests.size() > model_config.num_mshrs) {
            fake_stamp = get_shortest_stamp(ongoing_requests);
            process_ongoing_requests(ongoing_requests, fake_stamp);
        }

        //  Calculate reuse distance and access latency for each access
        int max_latency = -1;            //  max_latency of accesses int the same warp access
        int latency;
        for (i = 0; i < p_warp_access->size; i++) {
            int line_addr;
            int distance;

            line_addr = p_warp_access->accesses[i];

            //  get set id
            set_id = calculate_cache_set(line_addr, model_config);

            //  Calculate reuse distance and update stack
            distance = update_stack_tree(line_addr, Bs[set_id], Ps[set_id], set_counters[set_id]);

            //  Update the final output DistanceStat
            stat.increase(p_warp_access->pc, distance);

            //  Calculate latency of this access
            if (distance < model_config.cache_way_size) {
                latency = 0;

                //  In the case of pending hit, the latency is not zero
                std::multimap<int, int>::iterator it;
                it = ongoing_requests.find(line_addr);
                if (it != ongoing_requests.end()) {
                    latency = it->second - fake_stamp;
                }
            }
            else {
                latency = normal_generator.next_number();
            }

            //  Update max_latency
            if (latency > max_latency)
                max_latency = latency;

            //  Use the latency to update ongoing_requests
            if (latency > 0)
                ongoing_requests.insert(line_addr, fake_stamp + latency);

        }

        //  Use the max_latency to update warp_trace jam info
        if (model_config.jam_instruction) {
            if (p_warp_access->jam) {
                task.set_last_warptrace_jam(fake_stamp + max_latency);
            }
        }

        //  Increast fake_stamp
        fake_stamp ++;
    }

}

void process_ongoing_requests(std::multimap<int, int> & ongoing_requests, int fake_stamp) {
    std::multimap<int, int>::iterator it;

    it = ongoing_requests.begin();
    while (it != ongoing_requests.end()) {
        if (it->second <= fake_stamp) {
            it = ongoing_requests.erase(it);
        }
        else {
            it ++;
        }
    }
}

int get_shortest_stamp(std::multimap<int, int> & ongoing_requests) {
    int t;

    std::multimap<int, int>::iterator it;
    for (it = ongoing_requests.begin(); it != ongoing_requests.end(); it++) {
        if (it == ongoing_requests.begin())
            t = it->second;
        if (it->second < t)
            t = it->second;
    }

    return t;
}

//  update_stack_tree
//  update reusedistance stack
//  Returns reuse distance of the line address access
int update_stack_tree(addr_type line_addr, Tree & B, std::map<addr_type, int> & P, int & set_counter) {
    int previous_time;
    int distance;

    //  Check last accurance of the same line address
    previous_time = -1;
    if (P.find(line_addr) != P.end())
        previous_time = P[line_addr];

    //  Calculate reuse distance
    distance = -1;
    if (previous_time >= 0)
        distance = B.count(previous_time);

    //  Update P
    P[line_addr] = set_counter;

    //  Update B
    if (previous_time >= 0)
        B.unset(previous_time);
    B.set(set_counter);

    //  Increase set_counter
    set_counter ++;

    //  Reutrn reuse distance
    return distance;
}
