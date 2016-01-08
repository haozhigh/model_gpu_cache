


#include "AnalyseTask.h"


AnalyseTask::AnalyseTask() {
    location = 0;
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

void AnalyseTask::reset() {
    int i;

    location = 0;
    for (i = 0; i < p_warp_traces.size(); i++)
        p_warp_traces[i]->reset();
}

int AnalyseTask::next_available_warp_trace(int time_stamp) {
    int i;
    int n;

    i = location;
    n = p_warp_traces.size();
    while (true) {
        if (p_warp_traces[i]->is_available(time_stamp))
            return i;
        i = (i + 1) % n;
        if (i == location)
            return -1;
    }
}

bool AnalyseTask::is_finish() {
    int i;

    for (i = 0; i < p_warp_traces.size(); i++) {
        if (! p_warp_traces[i]->is_finish())
            return false;
    }

    return true;
}

WarpAccess * AnalyseTask::next_warp_access(int time_stamp) {
    int new_location;

    new_location = this->next_available_warp_trace(time_stamp);
    if (new_location < 0)
        return NULL;

    location = (new_location + 1) % p_warp_traces.size();
    return p_warp_traces[new_location]->next_warp_access(time_stamp);
}

void AnalyseTask::set_last_warptrace_jam(int time_stamp) {
    int last_location;

    last_location = (location + p_warp_traces.size() - 1) % p_warp_traces.size();
    p_warp_traces[last_location]->set_jam(time_stamp);
    return;
}
