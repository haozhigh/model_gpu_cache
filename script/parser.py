#!/usr/bin/python3

import os.path as path
from pandas import Series

from common import *


##  Break down a wide kernel name to (suite, bench, kernel)
def breakdown_wide_kernel_name(wide_kernel_name):
    break_kernel_name = wide_kernel_name.split("#")
    return (break_kernel_name[0], break_kernel_name[1], break_kernel_name[2])

##  Read text file content as a whole str
def read_text_file(file_path):
    ##  Check if the file exists
    if (path.isfile(file_path)):
        ##  try if the file can be opened
        try:
            file_handle = open(file_path, 'r')
        ##  If the file cannot be opened
        except IOError:
            print("####  read_text_file: can not open file for read :'" + file_path + "'  ####")
            return ""
        ##  If the file was opened successfully
        else:
            content = file_handle.read()
            file_handle.close()
            return content

    ##  If the file does not exist
    else:
        print("####  read_text_file: file does not exist :'" + file_path + "'  ####")
        return ""

##  Get wide kennel names for the gen trace output dir
##  Wide kernel names have the form: "suite#bench#kernel_name"
def get_wide_kernel_names_trace(trace_out_dir):
    wide_kernel_names = list()

    ##  Get suite names in the trace output dir
    suites = os.listdir(trace_out_dir)
    for suite in suites:

        ##  Get bench name in the trace output dir for a specific suite
        trace_out_dir_suite = path.join(trace_out_dir, suite)
        benches = os.listdir(trace_out_dir_suite)
        for bench in benches:

            ##  Get file names in the trace output dir for a specific suite/bench
            trace_out_dir_bench = path.join(trace_out_dir_suite, bench)
            bench_out_files = os.listdir(trace_out_dir_bench)
            for bench_out_file in bench_out_files:

                ##  Check if the file ends with .trc, if so, a new kernel is found
                if bench_out_file.endswith(".trc"):
                    wide_kernel_names.append(suite + "#" + bench + "#" + bench_out_file[:-4])

    ##  Sort all the wide kernel names
    wide_kernel_names.sort();

    ##  Return the width kernel names
    return wide_kernel_names

def parse_model_out(wide_kernel_names, model_out_dir):
    model_comp_miss = Series(0.0, index = wide_kernel_names)
    model_uncomp_miss = Series(0.0, index = wide_kernel_names)

    ##  Iterat over all kernels
    for wide_kernel_name in wide_kernel_names:

        ##  Abstract info from wide_kernel_name
        (suite, bench, kernel) = breakdown_wide_kernel_name(wide_kernel_name)

        ##  Set model miss rate output file for this kernel
        out_file = path.join(model_out_dir, suite, bench, kernel + ".miss_rate")

        ##  Check if the out_file exists
        if os.path.isfile(out_file):

            ##  Read miss rate info from the file
            f_str = read_text_file(out_file)

            ##  Write miss rate to Series
            model_comp_miss[wide_kernel_name] = float(f_str.split(' ')[0])
            model_uncomp_miss[wide_kernel_name] = float(f_str.split(' ')[1])

    ##  Return the Series
    return (model_comp_miss, model_uncomp_miss)

def parse_profiler_out(wide_kernel_names, profiler_out_dir):
    profiler_miss = Series(0.0, index = wide_kernel_names)

    ##  Iterate over all kernels
    for wide_kernel_name in wide_kernel_names:

        ##  Breakdown wide kernel name to (suite, bench, kernel)
        (suite, bench, kernel) = breakdown_wide_kernel_name(wide_kernel_name)

        ##  Set profiler miss rate output file for this kernel
        out_file = path.join(profiler_out_dir, suite, bench + ".prof")
        
        ##  Check if the out_file exists
        if os.path.isfile(out_file):
            
            ##  Read file content
            f_str = read_text_file(out_file)
            
            ##  Parse the miss rate for this kernel
            ##  Set pattern1 to find the occurance of this kernel name
            pattern1 = re.compile(r'Kernel:\s+' + kernel)
            match1 = pattern1.search(f_str)
            
            ##  If pattern1 is not found, set miss rate to 0
            if match1 == None:
                print("####  parse_profiler_out:: match1 is None, set miss rate to 0  ####")
                profiler_miss[wide_kernel_name] = 0
            else:

                ##  Set pattern2 to find the l1 cache avg miss rate after the end of match1
                pattern2 = re.compile(r'l1_cache_global_hit_rate\s+L1 Global Hit Rate\s+\S+%\s+\S+%\s+(\S+)%')
                match2 = pattern2.search(f_str, pos = match1.end())

                ##  If pattern2 is not found, set miss rate to 0
                if match2 == None:
                    print("####  parse_profiler_out:: match2 is None, set miss rate to 0  ####")
                    profiler_miss[wide_kernel_name] = 0
                else:
                    ##  Get miss rate
                    profiler_miss[wide_kernel_name] = 1 - float(match2.group(1)) / 100.0

    ##  Return the Series
    return profiler_miss

def parse_sim_out(wide_kernel_names, sim_out_dir):
    sim_miss = Series(0.0, index = wide_kernel_names)

    ##  Iterate over all kernels
    for wide_kernel_name in wide_kernel_names:

        ##  Breakdown wide kernel name to (suite, bench, kernel)
        (suite, bench, kernel) = breakdown_wide_kernel_name(wide_kernel_name)

        ##  Set profiler miss rate output file for this kernel
        out_file = path.join(profiler_out_dir, suite, bench + ".simlog")
        
        ##  Check if the out_file exists
        if os.path.isfile(out_file):
            
            ##  Read file content
            f_str = read_text_file(out_file)
 




if __name__ == '__main__':
    print(get_wide_kernel_names_trace())
