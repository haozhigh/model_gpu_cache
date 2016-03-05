



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
    std::multimap<addr_type, int> ongoing_requests;

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
    Ps.resize(model_config.cache_set_size);
	
    //  Record processed number of accesses per cache set
    std::vector<int> set_counters;
    set_counters.resize(model_config.cache_set_size);
    for (i = 0; i < set_counters.size(); i++)
        set_counters[i] = 0;

	
    //  Start the main loop
    fake_stamp = 0;
    task.reset();
    while (! task.is_finish()) {
        //  Process ongoing requests at eache new time stamp
        process_ongoing_requests(ongoing_requests, fake_stamp, Bs, Ps, set_counters, model_config);

        //  If all warps are jamed, get a NULL pointer, and thuns increase fake_stamp and continue
        p_warp_access = task.next_warp_access(fake_stamp);
        if (p_warp_access == NULL) {
            fake_stamp ++;
            continue;
        }

        //  Take a warptrace
        //  If MSHR check fails
        //  Increast fake_stamp to a point where more mshrs will be available
        while (p_warp_access->size + ongoing_requests.size() > model_config.num_mshrs) {
            fake_stamp = get_shortest_stamp(ongoing_requests);
            process_ongoing_requests(ongoing_requests, fake_stamp, Bs, Ps, set_counters, model_config);
        }

        //  Calculate reuse distance and access latency for each access
        int max_latency = -1;            //  max_latency of accesses int the same warp access
        int latency;
        for (i = 0; i < p_warp_access->size; i++) {
            addr_type line_addr;
            int distance;

            line_addr = p_warp_access->accesses[i];

            //  If the model is configed allocate_on_miss,
            //  Calculate reuse distance, and
            //  update the stack immediately, no matter it's a hit or miss
            if (model_config.allocate_on_miss) {
                distance = calculate_reuse_distance_update_stack_tree(line_addr, Bs, Ps, set_counters, model_config);
            }
            else {
                //  else, just calculate the reuse distance for now
                distance = calculate_reuse_distance(line_addr, Bs, Ps, model_config);
            }

            //  Update the final output DistanceStat
            stat.increase(p_warp_access->pc, distance);

            //  Calculate latency of this access
            if (distance < model_config.cache_way_size && distance >= 0) {
            //  if (distance < model_config.cache_way_size) {
                latency = 0;

                //  If the model is configed ad not allocate on miss,
                //  put the access to stack now for a cache hit
                if (! model_config.allocate_on_miss) {
                    update_stack_tree(line_addr, Bs, Ps, set_counters, model_config);
                }

                //  In the case of pending hit, the latency is not zero
                //  Pending hit only happens in allocate_on_miss config
                if (model_config.allocate_on_miss) {
                    std::multimap<addr_type, int>::iterator it;
                    it = ongoing_requests.find(line_addr);
                    if (it != ongoing_requests.end()) {
                        latency = it->second - fake_stamp;
                    }
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
                ongoing_requests.emplace(line_addr, fake_stamp + latency);

        }

        //  Use the max_latency to update warp_trace jam info
        if (model_config.jam_instruction) {
            if (p_warp_access->jam) {
                task.set_last_warptrace_jam(fake_stamp + max_latency);
            }
        }
        else {
            //  Original version: jam every access
            task.set_last_warptrace_jam(fake_stamp + max_latency);
        }

        //  Increast fake_stamp
        fake_stamp ++;
    }

}

void process_ongoing_requests(std::multimap<addr_type, int> & ongoing_requests,
                                                int fake_stamp,
                                                std::vector<Tree> & Bs,
                                                std::vector<std::map<addr_type, int>> & Ps,
                                                std::vector<int> & set_counters,
                                                ModelConfig & model_config) {
    std::multimap<addr_type, int>::iterator it;

    it = ongoing_requests.begin();
    while (it != ongoing_requests.end()) {
        if (it->second <= fake_stamp) {
            //  If the model is configed not allocate_on_miss, need to update stack tree here
            if (! model_config.allocate_on_miss) {
                update_stack_tree(it->first, Bs, Ps, set_counters, model_config);
            }

            it = ongoing_requests.erase(it);
        }
        else {
            it ++;
        }
    }
}

int get_shortest_stamp(std::multimap<addr_type, int> & ongoing_requests) {
    int t;

    std::multimap<addr_type, int>::iterator it;
    for (it = ongoing_requests.begin(); it != ongoing_requests.end(); it++) {
        if (it == ongoing_requests.begin())
            t = it->second;
        if (it->second < t)
            t = it->second;
    }

    return t;
}

int calculate_reuse_distance(addr_type line_addr,
                             std::vector<Tree> & Bs,
                             std::vector<std::map<addr_type, int>> & Ps,
                             ModelConfig & model_config) {
    int set_id;
    int distance;
    int previous_time;

    //  Calculate set_id
    set_id = calculate_cache_set(line_addr, model_config);

    //  Check last accurance of the same line address
    previous_time = -1;
    if (Ps[set_id].find(line_addr) != Ps[set_id].end())
        previous_time = Ps[set_id][line_addr];

    //  Calculate reuse distance
    distance = -1;
    if (previous_time >= 0)
        distance = Bs[set_id].count(previous_time);

    //  return distance
    return distance;
}

void update_stack_tree(addr_type line_addr,
                       std::vector<Tree> & Bs,
                       std::vector<std::map<addr_type, int>> & Ps,
                       std::vector<int> & set_counters,
                       ModelConfig & model_config) {
    int set_id;
    int previous_time;

    //  Calculate set_id
    set_id = calculate_cache_set(line_addr, model_config);

    //  Get previous time
    previous_time = -1;
    if (Ps[set_id].find(line_addr) != Ps[set_id].end())
        previous_time = Ps[set_id][line_addr];

    //  Update P
    Ps[set_id][line_addr] = set_counters[set_id];

    //  Update B
    if (previous_time >= 0)
        Bs[set_id].unset(previous_time);
    Bs[set_id].set(set_counters[set_id]);

    //  Increase the set_counter
    set_counters[set_id] ++;

    return;
}

int calculate_reuse_distance_update_stack_tree(addr_type line_addr,
                                               std::vector<Tree> & Bs,
                                               std::vector<std::map<addr_type, int>> & Ps,
                                               std::vector<int> & set_counters,
                                               ModelConfig & model_config) {
    int set_id;
    int previous_time;
    int distance;

    //  Calculate set_id
    set_id = calculate_cache_set(line_addr, model_config);

    //  Check last occurance of the same line address
    previous_time = -1;
    if (Ps[set_id].find(line_addr) != Ps[set_id].end())
        previous_time = Ps[set_id][line_addr];

    //  Calculate reuse distance
    distance = -1;
    if (previous_time >= 0)
        distance = Bs[set_id].count(previous_time);

    //  Update P
    Ps[set_id][line_addr] = set_counters[set_id];

    //  Update B
    if (previous_time >= 0)
        Bs[set_id].unset(previous_time);
    Bs[set_id].set(set_counters[set_id]);

    //  Increase set_counter
    set_counters[set_id] ++;

    //  Return distance
    return distance;
}
