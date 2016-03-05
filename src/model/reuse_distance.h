


#include "ModelConfig.h"
#include "AnalyseTask.h"
#include "DistanceStat.h"
#include "tree.h"




#ifndef MY_REUSE_DISTANCE
#define MY_REUSE_DISTANCE
    
void calculate_reuse_distance(AnalyseTask & task, ModelConfig & model_config, DistanceStat & stat);

int get_shortest_stamp(std::multimap<addr_type, int> & ongoing_requests);

void process_ongoing_requests(std::multimap<addr_type, int> & ongoing_requests,
        int fake_stamp,
        std::vector<Tree> & Bs,
        std::vector<std::map<addr_type, int>> & Ps,
        std::vector<int> & set_counters,
        ModelConfig & model_config);

int calculate_reuse_distance(addr_type line_addr,
                             std::vector<Tree> & Bs,
                             std::vector<std::map<addr_type, int>> & Ps,
                             ModelConfig & model_config);

void update_stack_tree(addr_type line_addr,
                       std::vector<Tree> & Bs,
                       std::vector<std::map<addr_type, int>> & Ps,
                       std::vector<int> & set_counters,
                       ModelConfig & model_config);

int calculate_reuse_distance_update_stack_tree(addr_type line_addr,
                                               std::vector<Tree> & Bs,
                                               std::vector<std::map<addr_type, int>> & Ps,
                                               std::vector<int> & set_counters,
                                               ModelConfig & model_config);
#endif
