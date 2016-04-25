#!/usr/bin/python3

import os.path as path


from pandas import Series, DataFrame
import pandas


from drawer import *

def main():
    ##  Read duration time from csv file
    duration_frame = pandas.read_csv(path.join(dir_script, "../output/duration.csv"))

    ##  Draw reuse distance calculation duration comparison
    draw_reuse_distance_duration_comparison_v2(duration_frame, "Reuse Distance Duration Comparison", path.join(dir_script, "../output/reuse_distance_duration_comparison.png"))

    ##  Add trace generation time to total model time
    #duration_frame['base_model'] = duration_frame['base_model'] + duration_frame['base_trace']
    #duration_frame['model'] = duration_frame['model'] + duration_frame['trace']

    duration_frame['model'] = duration_frame['opt_break_trace_off']

    ##  Call function to draw duration comparison
    draw_duration_v2(duration_frame, 'Time Consumption Comparison', path.join(dir_script, "../output/duration.png"))

if __name__ == '__main__':
    main()
