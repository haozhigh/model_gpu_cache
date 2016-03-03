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

    ##  Get all suite names
    ##  Draw a Error comparison chart for each suite
    draw_error_comparison(miss_frame, "Miss Rate Error Comparison", path.join(dir_script, "../output/miss_rate_error_compare.png"))


    ##  Draw cache miss break down to compulsory miss and uncompulsory miss
    draw_miss_breakdown(miss_frame, path.join(dir_script, "../output/miss_breakdown.png"))

    ##  Divide duration time by profiler duration for the same bench
    #duration_frame['base_model'] = duration_frame['base_model'] / duration_frame['profiler']
    #duration_frame['model'] = duration_frame['model'] / duration_frame['profiler']
    #duration_frame['base_trace'] = duration_frame['base_trace'] / duration_frame['profiler']
    #duration_frame['trace'] = duration_frame['trace'] / duration_frame['profiler']
    #duration_frame['sim'] = duration_frame['sim'] / duration_frame['profiler']




    ##  Read duration time from csv file
    duration_frame = pandas.read_csv(path.join(dir_script, "../output/duration.csv"))

    ##  Add trace generation time to total model time
    duration_frame['base_model'] = duration_frame['base_model'] + duration_frame['base_trace']
    duration_frame['model'] = duration_frame['model'] + duration_frame['trace']

    ##  Call function to draw duration comparison
    draw_duration(duration_frame, 'Time Consumption Comparison', path.join(dir_script, "../output/duration.png"))





    ##  Read memory footprint info fram csv file
    footprint_frame = pandas.read_csv(path.join(dir_script, "../output/footprint.csv"))

    ##  Draw memory footprint
    draw_footprint(footprint_frame, 'Kernel Memory Footprint', path.join(dir_script, "../output/footprint.png"))

    ##  Draw operation intensity
    draw_op_intensity(footprint_frame, 'Operation intensity for each kernel', path.join(dir_script, "../output/op_intensity.png"))

    ##  Draw miss rate
    draw_miss_rate(footprint_frame, 'L1 cache miss rate for each kernel', path.join(dir_script, "../output/miss_rate.png"))




    ##  Calculate even values
    model_miss_error = miss_frame['model_miss'] - miss_frame['profiler_miss']
    model_miss_error = model_miss_error.abs()
    base_model_miss_error = miss_frame['base_model_miss'] - miss_frame['profiler_miss']
    base_model_miss_error = base_model_miss_error.abs()
    sim_miss_error = miss_frame['sim_miss'] - miss_frame['profiler_miss']
    sim_miss_error = sim_miss_error.abs()

    print("model miss error mean : " + str(model_miss_error.mean()))
    print("base_model miss error mean : " + str(base_model_miss_error.mean()))
    print("sim miss error mean : " + str(sim_miss_error.mean()))

    model_duration = duration_frame['model'] + duration_frame['trace']
    base_model_duration = duration_frame['base_model'] + duration_frame['base_trace']
    sim_duration = duration_frame['sim']

    print("model duration mean: " + str(model_duration.mean()))
    print("base model duration mean: " + str(base_model_duration.mean()))
    print("sim duration mean: " + str(sim_duration.mean()))

    ##  Exclude trace generation time
    model_duration = duration_frame['model']
    base_model_duration = duration_frame['base_model']
    sim_duration = duration_frame['sim']

    print("model duration mean: " + str(model_duration.mean()))
    print("base model duration mean: " + str(base_model_duration.mean()))
    print("sim duration mean: " + str(sim_duration.mean()))



if __name__ == '__main__':
    main()
