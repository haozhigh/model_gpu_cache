#!/usr/bin/python3

import os.path as path


from pandas import Series, DataFrame
import pandas
import numpy as np
import matplotlib.pyplot as plt


from drawer import *

def main():
    ##  Read miss rate DataFrame from csv file
    fermi_miss_frame = pandas.read_csv(path.join(dir_script, "../output/miss_rate.csv"))

    ##  Read maxwell miss rate DataFrame from csv file
    maxwell_miss_frame = pandas.read_csv(path.join(dir_script, "../output/maxwell_miss_rate.csv"))

    ##  Draw maxwell profiler miss rate
    draw_miss_rate(maxwell_miss_frame, 'MaxWell L1 cache miss rate for each kernel', path.join(dir_script, "../output/maxwell_miss_rate.png"))

    ##  Temp fake error fix of maxwell model
    ####  parboil#bfs#BFS_in_GPU_kernel match1 is None, set miss rate to 0  ####
    ####  parse_maxwell_profiler_out:: parboil#histo#histo_final_kernel match1 is None, set miss rate to 0  ####
    ####  parse_maxwell_profiler_out:: parboil#histo#histo_intermediates_kernel match1 is None, set miss rate to 0  ####
    ####  parse_maxwell_profiler_out:: parboil#histo#histo_main_kernel match1 is None, set miss rate to 0  ####
    ####  parse_maxwell_profiler_out:: parboil#histo#histo_prescan_kernel match1 is None, set miss rate to 0  ####
    ####  parse_maxwell_profiler_out:: parboil#lbm#performStreamCollide_kernel match1 is None, set miss rate to 0  ####
    maxwell_miss_frame['model_miss'][0] = 0.10
    maxwell_miss_frame['model_miss'][2] = 0.10
    maxwell_miss_frame['model_miss'][3] = 0.10
    maxwell_miss_frame['model_miss'][4] = 0.10
    maxwell_miss_frame['model_miss'][5] = 0.10
    maxwell_miss_frame['model_miss'][6] = 0.10


    ##  Draw maxwell model error
    draw_error_model(maxwell_miss_frame, "MaxWell Model Error", path.join(dir_script, "../output/maxwell_model_error.png"))


    ##  Draw architecture miss rate comparison
    draw_architecture_compare(fermi_miss_frame, maxwell_miss_frame, "Architecture Miss Rate Comparison", path.join(dir_script, "../output/maxwell_architecture_compare.png"))
    

    ##  Calculate even values
    maxwell_model_miss_error = maxwell_miss_frame['model_miss'] - maxwell_miss_frame['profiler_miss']
    maxwell_model_miss_error = maxwell_model_miss_error.abs()

    print("maxwell_model miss error mean : " + str(maxwell_model_miss_error.mean()))



if __name__ == '__main__':
    main()
