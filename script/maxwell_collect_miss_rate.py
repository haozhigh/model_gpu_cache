#!/usr/bin/python3

import os.path as path


from pandas import Series, DataFrame
from parser import *




def main():
    ##  Set default trace out dir, and get wide kernel names
    trace_out_dir = path.join(dir_script, "../output/trace")
    wide_kernel_names = get_wide_kernel_names_trace(trace_out_dir)

    ##  
    maxwell_miss_frame = DataFrame(index = wide_kernel_names)

    ##  Set maxwell profiler out dir, and call the parsing function
    ##  Write maxwell profiler miss rate to the data frame
    maxwell_profiler_out_dir = path.join(dir_script, "../output/maxwell_profiler")
    maxwell_miss_frame['profiler_miss'] = parse_maxwell_profiler_out(wide_kernel_names, maxwell_profiler_out_dir)

    breakdown_frame_index_wide_kernel_name(maxwell_miss_frame)


    ##  Write to file
    maxwell_miss_out_file = path.join(dir_script, "../output/maxwell_miss_rate.csv")
    maxwell_miss_frame.to_csv(maxwell_miss_out_file)


if __name__ == '__main__':
    main()
