#!/usr/bin/python3

import os.path as path


from pandas import Series, DataFrame
from parser import *




def main():
    ##  Set default trace out dir, and get wide kernel names
    trace_out_dir = path.join(dir_script, "../output/trace")
    wide_kernel_names = get_wide_kernel_names_trace(trace_out_dir)

    ##  
    miss_frame = DataFrame(index = wide_kernel_names)

    ##  Set model out dir, and call the parsing function
    ##  Write model miss rate to the data frame
    model_out_dir = path.join(dir_script, "../output/trace")
    (model_comp_miss, model_uncomp_miss) = parse_model_out(wide_kernel_names, model_out_dir)
    miss_frame['model_comp_miss'] = model_comp_miss
    miss_frame['model_uncomp_miss'] = model_uncomp_miss
    miss_frame['model_miss'] = model_comp_miss + model_uncomp_miss

    ##  Set base model out dir, and call the parsing function
    ##  Write base model miss rate to the data frame
    base_model_out_dir = path.join(dir_script, "../output/base_trace")
    (base_model_comp_miss, base_model_uncomp_miss) = parse_base_model_out(wide_kernel_names, base_model_out_dir)
    miss_frame['base_model_comp_miss'] = base_model_comp_miss
    miss_frame['base_model_uncomp_miss'] = base_model_uncomp_miss
    miss_frame['base_model_miss'] = base_model_comp_miss + base_model_uncomp_miss

    ##  Set profiler out dir, and call the parsing function
    ##  Write profiler miss rate to the data frame
    profiler_out_dir = path.join(dir_script, "../output/profiler")
    miss_frame['profiler_miss'] = parse_profiler_out(wide_kernel_names, profiler_out_dir)

    ##  Set sim out dir, and call the parsing functioin
    ##  Write sim miss rate to the data frame
    sim_out_dir = path.join(dir_script, "../output/sim")
    miss_frame['sim_miss'] = parse_sim_out(wide_kernel_names, sim_out_dir)

    breakdown_frame_index(miss_frame)


    ##  Write to file
    miss_out_file = path.join(dir_script, "../output/miss_rate.csv")
    miss_frame.to_csv(miss_out_file)


    ##  Get wide bench names
    wide_bench_names = get_wide_bench_names(wide_kernel_names)

    duration_frame = DataFrame(index = wide_bench_names)
    duration_root_dir = path.join(dir_script, "../log")

    duration_frame['base_model'] = parse_duration_out(wide_bench_names, path.join(duration_root_dir, 'base_model'))
    duration_frame['model'] = parse_duration_out(wide_bench_names, path.join(duration_root_dir, 'model'))
    duration_frame['base_trace'] = parse_duration_out(wide_bench_names, path.join(duration_root_dir, 'base_trace'))
    duration_frame['trace'] = parse_duration_out(wide_bench_names, path.join(duration_root_dir, 'trace'))
    duration_frame['profiler'] = parse_duration_out(wide_bench_names, path.join(duration_root_dir, 'profiler'))
    duration_frame['sim'] = parse_duration_out(wide_bench_names, path.join(duration_root_dir, 'sim'))


    duration_out_file = path.join(dir_script, "../output/duration.csv")
    duration_frame.to_csv(duration_out_file)



if __name__ == '__main__':
    main()
