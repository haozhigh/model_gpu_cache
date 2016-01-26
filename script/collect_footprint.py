#!/usr/bin/python3

import os.path as path


from pandas import Series, DataFrame
from parser import *




def main():
    ##  Set default trace out dir, and get wide kernel names
    trace_out_dir = path.join(dir_script, "../output/trace")
    wide_kernel_names = get_wide_kernel_names_trace(trace_out_dir)





    footprint_frame = DataFrame(index = wide_kernel_names)

    ##  Set memory footprint out dir, and call the parsing function
    ##  Write footprint to footprint_frame
    footprint_out_dir = path.join(dir_script, "../log/profiler")
    footprint_frame['footprint'] = parse_footprint_out(wide_kernel_names, footprint_out_dir)





    ##  Set operation intensity out dir, and call the parsing function
    ##  Write operation intensity to op_intensity frame
    op_intensity_out_dir = path.join(dir_script, "../output/code")
    footprint_frame['op_intensity'] = parse_op_intensity_out(wide_kernel_names, op_intensity_out_dir)





    ##  Set profiler out dir, and call the parsing function
    ##  Write profiler miss rate to the data frame
    profiler_out_dir = path.join(dir_script, "../output/profiler")
    footprint_frame['profiler_miss'] = parse_profiler_out(wide_kernel_names, profiler_out_dir)





    ##  Break down frame index for wide kernel name
    breakdown_frame_index_wide_kernel_name(footprint_frame)

    ##  Write to file
    footprint_out_file = path.join(dir_script, "../output/footprint.csv")
    footprint_frame.to_csv(footprint_out_file)


if __name__ == '__main__':
    main()
