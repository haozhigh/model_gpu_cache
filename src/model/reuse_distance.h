


#include "ModelConfig.h"
#include "AnalyseTask.h"
#include "DistanceStat.h"
#include "tree.h"




#ifndef MY_REUSE_DISTANCE
#define MY_REUSE_DISTANCE
    
void calculate_reuse_distance(AnalyseTask & task, ModelConfig & model_config, DistanceStat & stat);

void process_ongoing_requests(std::multimap<int, int> & ongoing_requests, int fake_stamp);

int get_shortest_stamp(std::multimap<int, int> & ongoing_requests);

int update_stack_tree(addr_type line_addr, Tree & B, std::map<addr_type, int> & P, int & set_counter);

#endif
