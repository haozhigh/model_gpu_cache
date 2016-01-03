


#include <map>


#ifndef MY_DISTANCE_STAT
#define MY_DISTANCE_STAT

class DistanceStat {
    private:
    std::map<int, std::map<int, int>> stat;

    public:
    DistanceStat();

    void merge(DistanceStat & stat2);
    void write_to_file(std::string file_path);
};



#endif
