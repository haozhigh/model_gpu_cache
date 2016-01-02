

#include <ocelot/trace/interface/TraceEvent.h>

#include "MyWarpAccess.h"

#ifndef MY_LAST_LOAD
#define MY_LAST_LOAD

class MyLastLoad {
    private:
        MyWarpAccess *warp_accesses;
        int block_size;
        int num_warps_per_block;
        int block_id;

        int strip_reg_number(const std::string str);

    public:
        MyLastLoad();
        void update(const trace::TraceEvent &event);
        void check_jam(const trace::TraceEvent &event);
        void assign_memory(int b_size);
        void release_memory();
        void write_to_file(std::ofstream &out_stream);
        void write_to_file(std::ofstream &out_stream, const trace::TraceEvent &event);
        ~MyLastLoad();
};

#endif
