#!/usr/bin/python3

import os
import re
import os.path as path
from pandas import Series



dir_script = path.dirname(path.abspath(__file__))


def demangle_cpp_fun_name(name):
    cmd = "c++filt " + name
    pipe = os.popen(cmd)
    data = pipe.read()
    pipe.close()
    return data.split('(')[0]



##  Break down a wide kernel name to (suite, bench, kernel)
def breakdown_wide_kernel_name(wide_kernel_name):
    break_kernel_name = wide_kernel_name.split("#")
    return (break_kernel_name[0], break_kernel_name[1], break_kernel_name[2])
    
##  Break down a wide bench name to (suite, bench)
def breakdown_wide_bench_name(wide_bench_name):
    break_bench_name = wide_bench_name.split("#")
    return (break_bench_name[0], break_bench_name[1])

##  Get wide bench names from wide kernel names
def get_wide_bench_names(wide_kernel_names):
    wide_bench_names_set = set()
    for wide_kernel_name in wide_kernel_names:
        (suite, bench, kernel) = breakdown_wide_kernel_name(wide_kernel_name)
        wide_bench_names_set.add(suite + "#" + bench)
    
    wide_bench_names_list = list(wide_bench_names_set)
    wide_bench_names_list.sort()
    return wide_bench_names_list

def breakdown_frame_index_wide_bench_name(data_frame):
    suites = Series("", index = data_frame.index)
    benches = Series("", index = data_frame.index)

    for wide_bench_name in data_frame.index:
        (suite, bench) = breakdown_wide_bench_name(wide_bench_name)
        suites[wide_bench_name] = suite
        benches[wide_bench_name] = bench

    data_frame['suite'] = suites
    data_frame['bench'] = benches

def breakdown_frame_index_wide_kernel_name(data_frame):
    suites = Series("", index = data_frame.index)
    benches = Series("", index = data_frame.index)
    kernels = Series("", index = data_frame.index)

    for wide_kernel_name in data_frame.index:
        (suite, bench, kernel) = breakdown_wide_kernel_name(wide_kernel_name)
        suites[wide_kernel_name] = suite
        benches[wide_kernel_name] = bench
        kernels[wide_kernel_name] = kernel


    data_frame['suite'] = suites
    data_frame['bench'] = benches
    data_frame['kernel'] = kernels



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

def parse_base_model_out(wide_kernel_names, base_model_out_dir):
    base_model_comp_miss = Series(0.0, index = wide_kernel_names)
    base_model_uncomp_miss = Series(0.0, index = wide_kernel_names)

    ##  Iterate over all kernels
    for wide_kernel_name in wide_kernel_names:

        ##  Break down wide kerenl name
        (suite, bench, kernel) = breakdown_wide_kernel_name(wide_kernel_name)

        ##  Set base model miss rate output file for this kernel
        out_file = path.join(base_model_out_dir, suite, bench, kernel + '.distance')

        ##  Check if the out file exists
        if path.isfile(out_file):

            ##  Read file content
            f_str = read_text_file(out_file)

            ##  Parse file content
            access_count = 0
            comp_miss_count = 0
            hit_count = 0

            p = re.compile(r'^modelled_accesses: (\d*)\s*$', re.MULTILINE)
            m = p.search(f_str)
            if m == None:
                access_count = 0
            else:
                access_count = int(m.group(1))

            p = re.compile(r'^modelled_misses\(compulsory\): (\d*)\s*$', re.MULTILINE)
            m = p.search(f_str)
            if m == None:
                comp_miss_count = 0
            else:
                comp_miss_count = int(m.group(1))

            p = re.compile(r'^modelled_hits: (\d*)\s*$', re.MULTILINE)
            m = p.search(f_str)
            if m == None:
                hit_count = 0
            else:
                hit_count = int(m.group(1))

            base_model_comp_miss[wide_kernel_name] = float(comp_miss_count) / float(access_count)
            base_model_uncomp_miss[wide_kernel_name] = 1 - float(comp_miss_count + hit_count) / float(access_count)

    ##  Return
    return (base_model_comp_miss, base_model_uncomp_miss)
    

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

    suites = os.listdir(sim_out_dir)
    for suite in suites:

        sim_out_dir_suite = path.join(sim_out_dir, suite)
        bench_files = os.listdir(sim_out_dir_suite)
        for bench_file in bench_files:

            bench = bench_file.split('.')[0]

            f_str = read_text_file(path.join(sim_out_dir_suite, bench_file))

            sum_hit = 0
            sum_hit_reserved = 0
            sum_miss = 0
            sum_reservation_fail = 0

            p1 = re.compile(r'^kernel_name = (\S*)\s*$', re.MULTILINE)
            p2 = re.compile(r'^\s*Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\] = (\d*)\s*$', re.MULTILINE)
            p3 = re.compile(r'^\s*Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT_RESERVED\] = (\d*)\s*$', re.MULTILINE)
            p4 = re.compile(r'^\s*Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[MISS\] = (\d*)\s*$', re.MULTILINE)
            p5 = re.compile(r'^\s*Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[RESERVATION_FAIL\] = (\d*)\s*$', re.MULTILINE)
            m1 = p1.search(f_str)
            while m1 != None:
                kernel = demangle_cpp_fun_name(m1.group(1))
                hit_count = sum_hit
                hit_reserved_count = sum_hit_reserved
                miss_count = sum_miss
                reservation_fail_count = sum_reservation_fail

                m2 = p2.search(f_str, pos = m1.end())
                if m2 != None:
                    hit_count = int(m2.group(1))
                hit_count = hit_count - sum_hit
                sum_hit = sum_hit + hit_count

                m3 = p3.search(f_str, pos = m1.end())
                if m3 != None:
                    hit_reserved_count = int(m3.group(1))
                hit_reserved_count = hit_reserved_count - sum_hit_reserved
                sum_hit_reserved = sum_hit_reserved + hit_reserved_count
        
                m4 = p4.search(f_str, pos = m1.end())
                if m4 != None:
                    miss_count = int(m4.group(1))
                miss_count = miss_count - sum_miss
                sum_miss = sum_miss + miss_count

                m5 = p5.search(f_str, pos = m1.end())
                if m5 != None:
                    reservation_fail_count = int(m5.group(1))
                reservation_fail_count = reservation_fail_count - sum_reservation_fail
                sum_reservation_fail = sum_reservation_fail + reservation_fail_count

                access_count = hit_count + hit_reserved_count + miss_count

                try_wide_kernel_name = suite + '#' + bench + '#' + kernel
                if try_wide_kernel_name in wide_kernel_names:
                    sim_miss[try_wide_kernel_name] = float(miss_count) / float(access_count)

                m1 = p1.search(f_str, pos = m1.end())

    return sim_miss


def parse_duration_out(wide_bench_names, duration_out_dir):
    durations = Series(0, index = wide_bench_names)

    ##  Iterate over all kernels
    for wide_bench_name in wide_bench_names:

        ##  Breakdown wide kernel name to (suite, bench, kernel)
        (suite, bench) = breakdown_wide_bench_name(wide_bench_name)

        ##  Set profiler miss rate output file for this kernel
        out_file = path.join(duration_out_dir, suite, bench + ".duration")
        
        ##  Check if the out_file exists
        if os.path.isfile(out_file):
            
            ##  Read file content
            f_str = read_text_file(out_file)

            ##  Parse duration
            durations[wide_bench_name] = int(f_str)

    return durations

 
 




if __name__ == '__main__':
    print(get_wide_kernel_names_trace())
