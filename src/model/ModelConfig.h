

#ifndef MY_MODEL_CONFIG
#define MY_MODEL_CONFIG


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
    
    int num_sms;
    int max_active_threads;
    int max_active_blocks;

    int num_running_threads;
    
    int mapping_type;
    int coalescing_type;

    int mshr_check;
    int num_mshrs;

    int cache_line_bits;

    public:
    void calculate_line_bits();
    void print();
};


#endif
