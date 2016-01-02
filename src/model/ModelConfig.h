

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

    int cache_line_bits;

    public:
    void calculate_line_bits();
    void print();
};


#endif
