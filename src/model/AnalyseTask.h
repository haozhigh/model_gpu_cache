


#include "Access.h"
#include "ModelConfig.h"



#ifndef MY_ANALYSE_TASK
#define MY_ANALYSE_TASK

class AnalyseTask {
    private:
    std::vector<WarpTrace *> p_warp_traces;

    public:
    AnalyseTask();

    void add_warp_trace(WarpTrace * _p_warp_trace);

    static void generate_tasks(std::vector<WarpTrace> & warp_traces, std::vector<AnalyseTask> & tasks, ModelConfig & model_config, ThreadDim & thread_dim);
};


#endif
