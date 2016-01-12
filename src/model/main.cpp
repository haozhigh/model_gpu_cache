

#include <pthread.h>

#include <iostream>
#include <vector>

#include "functions.h"
#include "ModelConfig.h"
#include "Access.h"
#include "io.h"
#include "AnalyseTask.h"
#include "DistanceStat.h"
#include "reuse_distance.h"

class ThreadArgu {
    public:
    int num_threads;
    int thread_id;
    std::vector<AnalyseTask> * p_tasks;
    ModelConfig * p_model_config;
    DistanceStat * p_stat;
};

void *thread_task(void * _thread_argu) {
    int task_id;
    ThreadArgu * thread_argu;

    thread_argu = (ThreadArgu *)_thread_argu;

    for (task_id = thread_argu->thread_id; task_id < thread_argu->p_tasks->size(); task_id += thread_argu->num_threads) {
        calculate_reuse_distance(thread_argu->p_tasks->at(task_id), *(thread_argu->p_model_config), *(thread_argu->p_stat));
    }

    pthread_exit(NULL);
}


int main(int argc, char **argv) {
    ModelConfig model_config;
    ThreadDim thread_dim;
    std::vector<WarpTrace> warp_traces;
    std::vector<AnalyseTask> tasks;
    std::vector<DistanceStat> stats;


    //  Number of arguments check
    //  Argument 0: executable file name
    //  Argument 1: input trace file path
    //  Argument 2: output file path
    //  Argument 3: config file path
    if (argc != 4) {
        std::cout << "####  main: Too many or too few arguments.  ####" << std::endl;
        return -1;
    }

    //  Read model config from coresponding file
    std::cout << "####  main: Reading model config from '" << argv[3] << "'  ####" << std::endl;
    read_model_config_from_file(argv[3], model_config);

    //  Read input trace from file
    //  Coalescing is already done in this phase
    std::cout << "####  main: Reading trace from '" << argv[1] << "'  ####" << std::endl;
    read_trace_from_file(argv[1], warp_traces, thread_dim, model_config);

    //  Generate tasks
    AnalyseTask::generate_tasks(warp_traces, tasks, model_config, thread_dim);

    //  Assign memory for stats
    stats.resize(model_config.num_running_threads);

    //  Launch multiple threads to do the work
    std::vector<pthread_t> threads;
    std::vector<ThreadArgu> thread_argus;

    threads.resize(model_config.num_running_threads);
    thread_argus.resize(model_config.num_running_threads);
    stats.resize(model_config.num_running_threads);
    for (int i = 0; i < model_config.num_running_threads; i++) {
        int err;

        thread_argus[i].num_threads = model_config.num_running_threads;
        thread_argus[i].thread_id = i;
        thread_argus[i].p_tasks = &tasks;
        thread_argus[i].p_model_config = &model_config;
        thread_argus[i].p_stat = &stats[i];

        err = pthread_create(&threads[i], NULL, thread_task, (void *)&thread_argus[i]);
        if (err != 0) {
            std::cout << "main: Failed to create thread " << i << std::endl;
            return -1;
        }
    }

    //  Wait for all the work thead to exit
    for (int i = 0; i < model_config.num_running_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    //  Combine the DistanceStat from different threads
    for (int i = 1; i < model_config.num_running_threads; i++) {
        stats[0].merge(stats[i]);
    }

    //  Write reuse distance stat to file
    std::cout << "####  main: Writing distances to '" << argv[2] << "'  ####" << std::endl;
    stats[0].write_to_file(argv[2], model_config);

    return 0;
}
