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

    ##  Draw maxwell model error
    draw_error_model_v2(maxwell_miss_frame, "MaxWell Model Error", path.join(dir_script, "../output/maxwell_model_error.png"))


    ##  Draw architecture miss rate comparison
    draw_architecture_compare_v2(fermi_miss_frame, maxwell_miss_frame, "Architecture Miss Rate Comparison", path.join(dir_script, "../output/maxwell_architecture_compare.png"))

    ##  Draw architecture model miss rate comparison
    draw_architecture_model_compare_v2(fermi_miss_frame, maxwell_miss_frame, "Architecture Model Miss Rate Comparison", path.join(dir_script, "../output/maxwell_architecture_model_compare.png"))

    ##  Draw architecture miss rate error comparison
    draw_architecture_error_compare_v2(fermi_miss_frame, maxwell_miss_frame, "Architecture Miss Rate Comparison", path.join(dir_script, "../output/maxwell_architecture_error_compare.png"))
    

    ##  Calculate even values
    maxwell_model_miss_error = maxwell_miss_frame['model_miss'] - maxwell_miss_frame['profiler_miss']
    maxwell_model_miss_error = maxwell_model_miss_error.abs()

    print("maxwell_model miss error mean : " + str(maxwell_model_miss_error.mean()))



if __name__ == '__main__':
    main()
