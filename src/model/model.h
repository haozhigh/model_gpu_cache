#include <string>






class ModelConfig {
	public:
	//  Basic cache size config
	int cache_line_size;
	int cache_way_size;
	int cache_set_size;

	int allocate_on_miss;
	int jam_instruction;

	int latency_type;
	int latency_mean;
	int latency_dev;

};

void read_model_config_from_file(std::string file_path, ModelConfig &model_config);
