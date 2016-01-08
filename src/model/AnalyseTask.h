


#include "Access.h"
#include "ModelConfig.h"



#ifndef MY_ANALYSE_TASK
#define MY_ANALYSE_TASK

class AnalyseTask {
    private:
    std::vector<WarpTrace *> p_warp_traces;
    int location;

    int next_available_warp_trace(int time_stamp);

    public:
    AnalyseTask();

    void add_warp_trace(WarpTrace * _p_warp_trace);

    static void generate_tasks(std::vector<WarpTrace> & warp_traces, std::vector<AnalyseTask> & tasks, ModelConfig & model_config, ThreadDim & thread_dim);

    void reset();
    bool is_finish();
    WarpAccess * next_warp_access(int time_stamp);
    void set_last_warptrace_jam(int time_stamp);
};


#endif
