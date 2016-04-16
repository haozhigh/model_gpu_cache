#!/usr/bin/python3

import os.path as path


from pandas import Series, DataFrame
import pandas


from drawer import *

def main():
    ##  Read memory footprint info fram csv file
    print("##  draw_footprint.py: Reading memory footprint from csv file. ##")
    footprint_frame = pandas.read_csv(path.join(dir_script, "../output/footprint.csv"))

    ##  Draw memory footprint
    print("##  draw_footprint.py: Drawing memory footprint. ##")
    draw_footprint_v2(footprint_frame, 'Kernel Memory Footprint', path.join(dir_script, "../output/footprint.png"))

    ##  Draw operation intensity
    print("##  draw_footprint.py: Drawing operation intensity. ##")
    draw_op_intensity_v2(footprint_frame, 'Operation intensity for each kernel', path.join(dir_script, "../output/op_intensity.png"))

    ##  Draw miss rate
    print("##  draw_footprint.py: Drawing miss rate. ##")
    draw_miss_rate_v2(footprint_frame, 'L1 cache miss rate for each kernel', path.join(dir_script, "../output/miss_rate.png"))

if __name__ == '__main__':
    main()
