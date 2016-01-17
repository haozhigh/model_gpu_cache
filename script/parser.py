#!/usr/bin/python3

import os.path as path
from pandas import Series

from common import *

##  Get wide kennel names for the gen trace output dir
##  Wide kernel names have the form: "suite#bench#kernel_name"
def get_wide_kernel_names_trace():
    wide_kernel_names = list()

    ##  Get suite names in the trace output dir
    dir_trace = path.join(dir_script, "../output/trace")
    suites = os.listdir(dir_trace)
    for suite in suites:

        ##  Get bench name in the trace output dir for a specific suite
        dir_trace_suite = path.join(dir_trace, suite)
        benches = os.listdir(dir_trace_suite)
        for bench in benches:

            ##  Get file names in the trace output dir for a specific suite/bench
            dir_trace_suite_bench = path.join(dir_trace_suite, bench)
            suite_files = os.listdir(dir_trace_suite_bench)
            for suite_file in suite_files:

                ##  Check if the file ends with .trc, if so, a new kernel is found
                if suite_file.endswith(".trc"):
                    wide_kernel_names.append(suite + "#" + bench + "#" + suite_file[:-4])

    ##  Sort all the wide kernel names
    wide_kernel_names.sort();

    ##  Return the width kernel names
    return wide_kernel_names

def parse_model_out(miss_frame, wide_kernel_names):
    model_comp_miss = Series(0.0, index = wide_kernel_names)
    model_uncomp_miss = Series(0.0, index = wide_kernel_names)

    ##  Iterat over all kernels
    for wide_kernel_name in wide_kernel_names:

        ##  Abstract info from wide_kernel_name
        suite = wide_kernel_name.split('#')[0]
        bench = wide_kernel_name.split('#')[1]
        kernel = wide_kernel_name.split('#')[2]

        ##  Set model miss rate output file for this kernel
        out_file = path.join(dir_script, "../output/trace", suite, bench, kernel + ".miss_rate")

        ##  Check if the out_file exists
        if os.path.isfile(out_file):

            ##  Read miss rate info from the file
            f_handle = open(out_file, 'r')
            f_str = f_handle.read()
            f_handle.close();

            ##  Write miss rate to Series
            model_comp_miss[wide_kernel_name] = float(f_str.split(' ')[0])
            model_uncomp_miss[wide_kernel_name] = float(f_str.split(' ')[1])

    ##  Write model miss rate to the data frame
    miss_frame['model_comp_miss'] = model_comp_miss
    miss_frame['model_uncomp_miss'] = model_uncomp_miss
    miss_frame['model_miss'] = model_comp_miss + model_uncomp_miss






if __name__ == '__main__':
    print(get_wide_kernel_names_trace())
