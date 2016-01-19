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

    ##  Set profiler out dir, and call the parsing function
    ##  Write profiler miss rate to the data frame
    profiler_out_dir = path.join(dir_script, "../output/profiler")
    miss_frame['profiler_miss'] = parse_profiler_out(wide_kernel_names, profiler_out_dir)




    print(miss_frame)



if __name__ == '__main__':
    main()
