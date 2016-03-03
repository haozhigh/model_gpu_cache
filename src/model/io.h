#include <string>
#include <vector>

#include "ModelConfig.h"
#include "Access.h"


#ifndef MY_IO
#define MY_IO

void read_model_config_from_file(std::string file_path, ModelConfig &model_config);


int read_trace_from_file(std::string file_path, std::vector<WarpTrace> &warp_traces, ThreadDim &thread_dim, const ModelConfig & model_config);

int read_thread_dim_from_file(std::string trace_path, ThreadDim &thread_dim);

int coalesce_addr(addr_type *addrs, int warp_size, int width);
int coalesce_addr_maxwell(addr_type *addrs, int warp_size, int width);

#endif
