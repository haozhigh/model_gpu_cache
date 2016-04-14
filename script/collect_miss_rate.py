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

    ##  Collect miss rate for opt_break_trace_off version
    opt_break_trace_off_out_dir = path.join(dir_script, "../output/opt_break/trace_off")
    (opt_break_trace_off_comp_miss, opt_break_trace_off_uncomp_miss) = parse_model_out(wide_kernel_names, opt_break_trace_off_out_dir)
    miss_frame['opt_break_trace_off_comp_miss'] = opt_break_trace_off_comp_miss
    miss_frame['opt_break_trace_off_uncomp_miss'] = opt_break_trace_off_uncomp_miss
    miss_frame['opt_break_trace_off_miss'] = opt_break_trace_off_comp_miss + opt_break_trace_off_uncomp_miss

    ##  Collect miss rate for opt_break_trace_on version
    opt_break_trace_on_out_dir = path.join(dir_script, "../output/opt_break/trace_on")
    (opt_break_trace_on_comp_miss, opt_break_trace_on_uncomp_miss) = parse_model_out(wide_kernel_names, opt_break_trace_on_out_dir)
    miss_frame['opt_break_trace_on_comp_miss'] = opt_break_trace_on_comp_miss
    miss_frame['opt_break_trace_on_uncomp_miss'] = opt_break_trace_on_uncomp_miss
    miss_frame['opt_break_trace_on_miss'] = opt_break_trace_on_comp_miss + opt_break_trace_on_uncomp_miss

    ##  Collect miss rate for opt_break_jam_off version
    opt_break_jam_off_out_dir = path.join(dir_script, "../output/opt_break/jam_off")
    (opt_break_jam_off_comp_miss, opt_break_jam_off_uncomp_miss) = parse_model_out(wide_kernel_names, opt_break_jam_off_out_dir)
    miss_frame['opt_break_jam_off_comp_miss'] = opt_break_jam_off_comp_miss
    miss_frame['opt_break_jam_off_uncomp_miss'] = opt_break_jam_off_uncomp_miss
    miss_frame['opt_break_jam_off_miss'] = opt_break_jam_off_comp_miss + opt_break_jam_off_uncomp_miss

    ##  Collect miss rate for opt_break_jam_on version
    opt_break_jam_on_out_dir = path.join(dir_script, "../output/opt_break/jam_on")
    (opt_break_jam_on_comp_miss, opt_break_jam_on_uncomp_miss) = parse_model_out(wide_kernel_names, opt_break_jam_on_out_dir)
    miss_frame['opt_break_jam_on_comp_miss'] = opt_break_jam_on_comp_miss
    miss_frame['opt_break_jam_on_uncomp_miss'] = opt_break_jam_on_uncomp_miss
    miss_frame['opt_break_jam_on_miss'] = opt_break_jam_on_comp_miss + opt_break_jam_on_uncomp_miss

    ##  Collect miss rate for opt_break_stack_off version
    opt_break_stack_off_out_dir = path.join(dir_script, "../output/opt_break/stack_off")
    (opt_break_stack_off_comp_miss, opt_break_stack_off_uncomp_miss) = parse_model_out(wide_kernel_names, opt_break_stack_off_out_dir)
    miss_frame['opt_break_stack_off_comp_miss'] = opt_break_stack_off_comp_miss
    miss_frame['opt_break_stack_off_uncomp_miss'] = opt_break_stack_off_uncomp_miss
    miss_frame['opt_break_stack_off_miss'] = opt_break_stack_off_comp_miss + opt_break_stack_off_uncomp_miss

    ##  Collect miss rate for opt_break_stack_on version
    opt_break_stack_on_out_dir = path.join(dir_script, "../output/opt_break/stack_on")
    (opt_break_stack_on_comp_miss, opt_break_stack_on_uncomp_miss) = parse_model_out(wide_kernel_names, opt_break_stack_on_out_dir)
    miss_frame['opt_break_stack_on_comp_miss'] = opt_break_stack_on_comp_miss
    miss_frame['opt_break_stack_on_uncomp_miss'] = opt_break_stack_on_uncomp_miss
    miss_frame['opt_break_stack_on_miss'] = opt_break_stack_on_comp_miss + opt_break_stack_on_uncomp_miss

    ##  Collect miss rate for opt_break_latency_off version
    opt_break_latency_off_out_dir = path.join(dir_script, "../output/opt_break/latency_off")
    (opt_break_latency_off_comp_miss, opt_break_latency_off_uncomp_miss) = parse_model_out(wide_kernel_names, opt_break_latency_off_out_dir)
    miss_frame['opt_break_latency_off_comp_miss'] = opt_break_latency_off_comp_miss
    miss_frame['opt_break_latency_off_uncomp_miss'] = opt_break_latency_off_uncomp_miss
    miss_frame['opt_break_latency_off_miss'] = opt_break_latency_off_comp_miss + opt_break_latency_off_uncomp_miss

    ##  Collect miss rate for opt_break_latency_on version
    opt_break_latency_on_out_dir = path.join(dir_script, "../output/opt_break/latency_on")
    (opt_break_latency_on_comp_miss, opt_break_latency_on_uncomp_miss) = parse_model_out(wide_kernel_names, opt_break_latency_on_out_dir)
    miss_frame['opt_break_latency_on_comp_miss'] = opt_break_latency_on_comp_miss
    miss_frame['opt_break_latency_on_uncomp_miss'] = opt_break_latency_on_uncomp_miss
    miss_frame['opt_break_latency_on_miss'] = opt_break_latency_on_comp_miss + opt_break_latency_on_uncomp_miss


    breakdown_frame_index_wide_kernel_name(miss_frame)


    ##  Write to file
    miss_out_file = path.join(dir_script, "../output/miss_rate.csv")
    miss_frame.to_csv(miss_out_file)

if __name__ == '__main__':
    main()
