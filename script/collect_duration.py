#!/usr/bin/python3

import os.path as path


from pandas import Series, DataFrame
from parser import *




def main():
    ##  Set default trace out dir, and get wide kernel names
    trace_out_dir = path.join(dir_script, "../output/trace")
    wide_kernel_names = get_wide_kernel_names_trace(trace_out_dir)

    ##  Get wide bench names
    wide_bench_names = get_wide_bench_names(wide_kernel_names)

    duration_frame = DataFrame(index = wide_bench_names)
    duration_root_dir = path.join(dir_script, "../log")

    duration_frame['base_model'] = parse_duration_out(wide_bench_names, path.join(duration_root_dir, 'base_model'))
    duration_frame['model'] = parse_duration_out(wide_bench_names, path.join(duration_root_dir, 'model'))
    duration_frame['model_compare'] = parse_duration_out(wide_bench_names, path.join(duration_root_dir, 'opt_break/trace_off'))
    duration_frame['base_trace'] = parse_duration_out(wide_bench_names, path.join(duration_root_dir, 'base_trace'))
    duration_frame['trace'] = parse_duration_out(wide_bench_names, path.join(duration_root_dir, 'trace'))
    duration_frame['profiler'] = parse_duration_out(wide_bench_names, path.join(duration_root_dir, 'profiler'))
    duration_frame['sim'] = parse_duration_out(wide_bench_names, path.join(duration_root_dir, 'sim'))

    breakdown_frame_index_wide_bench_name(duration_frame)

    duration_out_file = path.join(dir_script, "../output/duration.csv")
    duration_frame.to_csv(duration_out_file)



if __name__ == '__main__':
    main()
