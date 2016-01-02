

#include <fstream>

#include "constant.h"

#ifndef MY_WARP_ACCESS
#define MY_WARP_ACCESS

class MyAccess {
    public:
    unsigned long long address;

    MyAccess();
};

class MyWarpAccess {
    private:
    MyAccess accesses[WARP_SIZE];
    int global_warp_id;
    int pc;
    int target;
    int width;
    int jam;
    int valid;
    int num_valid_accesses;

    public:
    void write_to_file(std::ofstream &out_stream);
    void reset();
    void set_warp_info(int gwi, int p, int t, int width);
    void add(unsigned long long a);
    bool is_valid();
    bool is_jam();
    void check_jam(int reg_id);
    

    MyWarpAccess();
};

#endif
