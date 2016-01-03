


#include "AnalyseTask.h"


AnalyseTask::AnalyseTask() {

}

void AnalyseTask::add_warp_trace(WarpTrace * _p_warp_trace) {
    this->p_warp_traces.push_back(_p_warp_trace);
}

void AnalyseTask::generate_tasks(std::vector<WarpTrace> & warp_traces, std::vector<AnalyseTask> & tasks, ModelConfig & model_config, ThreadDim & thread_dim) {
    int active_blocks_per_sm;

    //  Calculate active_blocks_per_sm
    active_blocks_per_sm = model_config.max_active_threads / (thread_dim.threads_per_warp * thread_dim.warps_per_block);
    if (active_blocks_per_sm > model_config.max_active_blocks)
        active_blocks_per_sm = model_config.max_active_blocks;

    int warps_per_round;

    warps_per_round = model_config.num_sms * active_blocks_per_sm * thread_dim.warps_per_block;

    int tasks_per_round;
    tasks_per_round = model_config.num_sms;

    int warp_id;
    int task_id;
    int local_task_id;
    for (warp_id = 0; warp_id < warp_traces.size(); warp_id ++) {
        int round_id;
        int local_warp_id;

        round_id = warp_id / warps_per_round;
        local_warp_id = warp_id % warps_per_round;

        //  Calculate task id
        local_task_id = (local_warp_id / thread_dim.warps_per_block) % model_config.num_sms;
        task_id = local_task_id + round_id * tasks_per_round;

        //  Add this warp trace to the corresponding task
        if (task_id >= tasks.size())
            tasks.resize(task_id + 1);
        tasks[task_id].add_warp_trace(&warp_traces[warp_id]);
    }
}
