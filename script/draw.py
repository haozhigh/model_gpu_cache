#!/usr/bin/python3

import os.path as path


from pandas import Series, DataFrame
import pandas


from drawer import *

def main():
    ##  Read miss rate DataFrame from csv file
    miss_frame = pandas.read_csv(path.join(dir_script, "../output/miss_rate.csv"))

    ##  Get all suite names
    ##  Draw a Error comparison chart for each suite
    draw_error_comparison_v2(miss_frame, "Miss Rate Error Comparison", path.join(dir_script, "../output/miss_rate_error_compare.png"))


    ##  Draw cache miss break down to compulsory miss and uncompulsory miss
    draw_miss_breakdown(miss_frame, path.join(dir_script, "../output/miss_breakdown.png"))

if __name__ == '__main__':
    main()
