#!/usr/bin/python3

import os.path as path


from pandas import Series, DataFrame
import pandas
import numpy as np
import matplotlib.pyplot as plt


from drawer import *

def main():
    ##  Read miss rate DataFrame from csv file
    miss_frame = pandas.read_csv(path.join(dir_script, "../output/miss_rate.csv"))

    ##  Read duration time from csv file
    duration_frame = pandas.read_csv(path.join(dir_script, "../output/duration.csv"))

    ##  Add trace generation time to total model time
    duration_frame['base_model'] = duration_frame['base_model'] + duration_frame['base_trace']
    duration_frame['model'] = duration_frame['model'] + duration_frame['trace']



    ##  Calculate even errors and print out
    model_miss_error = miss_frame['model_miss'] - miss_frame['profiler_miss']
    model_miss_error = model_miss_error.abs()
    base_model_miss_error = miss_frame['base_model_miss'] - miss_frame['profiler_miss']
    base_model_miss_error = base_model_miss_error.abs()
    sim_miss_error = miss_frame['sim_miss'] - miss_frame['profiler_miss']
    sim_miss_error = sim_miss_error.abs()

    print("##  Even erros for model, base_model, and GPGPU-Sim  ##")
    print("model miss error mean : " + str(model_miss_error.mean()))
    print("base_model miss error mean : " + str(base_model_miss_error.mean()))
    print("sim miss error mean : " + str(sim_miss_error.mean()))
    print("")

    ##  Calculate even duration values and print out
    model_duration = duration_frame['model'] + duration_frame['trace']
    base_model_duration = duration_frame['base_model'] + duration_frame['base_trace']
    sim_duration = duration_frame['sim']

    print("##  Even duration for model, base_model, and GPGPU-Sim  ##")
    print("model duration mean: " + str(model_duration.mean()))
    print("base model duration mean: " + str(base_model_duration.mean()))
    print("sim duration mean: " + str(sim_duration.mean()))
    print("")

    ##  Calculate event duration values excluding trace generating time, and print out
    model_duration = duration_frame['model']
    base_model_duration = duration_frame['base_model']
    sim_duration = duration_frame['sim']

    print("##  Even duration for model, base_model, and GPGPU-Sim excluding trace generating time ##")
    print("model duration mean: " + str(model_duration.mean()))
    print("base model duration mean: " + str(base_model_duration.mean()))
    print("sim duration mean: " + str(sim_duration.mean()))
    print("")

    ##  Calculate even errors for opt_break versions
    opt_break_trace_off_miss_error = miss_frame['opt_break_trace_off_miss'] - miss_frame['profiler_miss']
    opt_break_trace_off_miss_error = opt_break_trace_off_miss_error.abs()
    opt_break_jam_off_miss_error = miss_frame['opt_break_jam_off_miss'] - miss_frame['profiler_miss']
    opt_break_jam_off_miss_error = opt_break_jam_off_miss_error.abs()
    opt_break_stack_off_miss_error = miss_frame['opt_break_stack_off_miss'] - miss_frame['profiler_miss']
    opt_break_stack_off_miss_error = opt_break_stack_off_miss_error.abs()
    opt_break_latency_off_miss_error = miss_frame['opt_break_latency_off_miss'] - miss_frame['profiler_miss']
    opt_break_latency_off_miss_error = opt_break_latency_off_miss_error.abs()

    print("##  Even erros for opt_break model versions  ##")
    print("opt_break_trace_off error mean : " + str(opt_break_trace_off_miss_error.mean()))
    print("opt_break_jam_off error mean : " + str(opt_break_jam_off_miss_error.mean()))
    print("opt_break_stack_off error mean : " + str(opt_break_stack_off_miss_error.mean()))
    print("opt_break_latency_off error mean : " + str(opt_break_latency_off_miss_error.mean()))
    print("")



if __name__ == '__main__':
    main()
