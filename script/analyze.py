#!/usr/bin/python3


import os
import sys
import os.path as path

from parser import *
from drawer import *


def main():

    ##  Check if there are proper commandline arguments
    ##  No whitespce within suite_names, bench_names or kernel_names
    if len(sys.argv) != 4:
        print(sys.argv)
        print("Commandline does not match 'analyze.py input_file output_path kernel_name'")
        return -1

    ##  Check if the distance output file exists for the kernel specified by arguments
    distance_file_path = sys.argv[1]
    out_dir_bench = sys.argv[2]
    kernel = sys.argv[3]
    if not path.isfile(distance_file_path):
        print("File does not exist: '" + distance_file_path + "'")
        return -1

    ##  Read file content
    file_content = read_text_file(distance_file_path)

    ##  Variables to record results
    distance_dict = dict()

    ##  Iterate over each line
    lines = file_content.split('\n')
    for line in lines:
        if len(line.strip()) < 1:
            continue
        str_nums = line.split(' ')
        pc_counter = int(str_nums[0])
        distance = int(str_nums[1])
        count = int(str_nums[2])

        ##  Update distance_dict
        distance_dict[distance] = distance_dict.get(distance, 0) + count


##  Convert distance_dict to distance_values and distance_counts sorted by distance_values
    distance_values = list()
    distance_counts = list()
    for i in sorted(distance_dict):
        distance_values.append(i)
        distance_counts.append(distance_dict[i])

##  Set the output image path for distance histogram
##  Call the function to do the drawing of distance histogram
    distance_histo_out_path = path.join(out_dir_bench, kernel + "_distance_histo.png")
    draw_distance_histo(distance_values, distance_counts, "Distance Histogram", distance_histo_out_path)




if __name__ == '__main__':
    main()
