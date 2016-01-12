


#include <map>
#include "ModelConfig.h"


#ifndef MY_DISTANCE_STAT
#define MY_DISTANCE_STAT

class DistanceStat {
    private:
    std::map<int, std::map<int, int>> stat;
    double compulsory_miss_rate;
    double uncompulsory_miss_rate;

    void calculate_miss_rate(const ModelConfig & model_config);
    void write_miss_rate_to_file(std::string file_path);

    public:
    DistanceStat();

    void increase(int pc, int distance);
    void merge(DistanceStat & stat2);
    void write_to_file(std::string file_path, const ModelConfig & model_config);
};



#endif
