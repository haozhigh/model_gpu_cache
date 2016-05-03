#!/usr/bin/python3


import os
import sys
import os.path as path

from pandas import Series, DataFrame
import pandas
import numpy as np

from parser import *
from drawer import *

##  Combine some big distance histo, return distance_values and distance_counts
def combine_distance_histo(distance_histo):
    ##  Varibales prepared to return, for drawing
    distance_values = list()
    distance_counts = list()

    ##  Combine count and limit
    number_count = 0
    number_limit = 14

    ##  Variable to record whether combine actually happened
    combine_happened = False

    ##  Iterate over each distance
    for distance in distance_histo.index:
        ##  If number limit no encoutered yet
        ##  Append distance_values and distance_counts
        if number_count <= number_limit:
            distance_values.append(str(distance))
            distance_counts.append(distance_histo[distance])
            number_count = number_count + 1
        ##  If number limit already encounted
        ##  Add current count to the last distance within limit
        else:
            distance_counts[number_count - 1] = distance_counts[number_count - 1] + distance_histo[distance]
            combine_happened = True

    ##  If combine actually happened, modify last ditance value to show that
    if combine_happened:
        #distance_values[number_count - 1] = ">=" + distance_values[number_count - 1]
        distance_values[number_count - 1] = distance_values[number_count - 1] + "+"

    return (distance_values, distance_counts)

##  Get a sorted list of distances in distance_records
def get_sorted_distances(distance_records):
    ##  Variable declare
    distances = set()

    ##  Iterate over each record
    for record in distance_records:
        distance = record[1]
        distances.add(distance)
    
    ##  Convert distances from set to list and sort itself
    distances_list = list(distances)
    distances_list.sort()

    ##  Return the sorted distances
    return distances_list

##  Get a sorted list of pcs in distance_records
def get_sorted_pcs(distance_records):
    ##  Varibale declare
    pcs = set()

    ##  Iterate over each record
    for record in distance_records:
        pc = record[0]
        pcs.add(pc)

    ##  Convert distances from set to list and sort itself
    pcs_list = list(pcs)
    pcs_list.sort()

    ##  Return the sorted distances
    return pcs_list

##  Do miss type analysis
def miss_type_analyze(distance_records, out_file_path):
    hit_count = 0
    comp_miss_count = 0
    uncomp_miss_count = 0

    ##  Iterate over each record
    for record in distance_records:
        distance = record[1]
        count = record[2]
        if distance < 0:
            comp_miss_count = comp_miss_count + count
        else:
            if distance < 4:
                hit_count = hit_count + count
            else:
                uncomp_miss_count = uncomp_miss_count + count

    ##  Call the function to do drawing
    draw_miss_type(hit_count, comp_miss_count, uncomp_miss_count, out_file_path)

##  Do distance histo analsis
def distance_histo_analyze(distance_records, out_file_path):
    ##  Get sorted distances
    distances = get_sorted_distances(distance_records)

    ##  Define a Series to store distance histo
    distance_histo = Series(0, index = distances)

    ##  Iterate over each record
    for record in distance_records:
        distance = record[1]
        count = record[2]
        distance_histo[distance] = distance_histo[distance] + count
     
    ##  Call the function to do distance combine
    (distance_values, distance_counts) = combine_distance_histo(distance_histo)

    ##  Call the function to do drawing
    draw_distance_histo(distance_values, distance_counts, out_file_path)

##  Do pc histo analysis
def pc_histo_analyze(distance_records, out_file_path):
    ##  Set cache_way_size
    cache_way_size = 4

    ##  Get sorted pcs
    pcs = get_sorted_pcs(distance_records)

    ##  Define 2 Serieses to store pc histo
    pc_hit_histo = Series(0, index = pcs)
    pc_miss_histo = Series(0, index = pcs)

    ##  Iterate over each record
    for record in distance_records:
        pc = record[0]
        distance = record[1]
        count = record[2]
        
        ##  Eaxm if it is hit or miss
        if distance < cache_way_size:
            if distance == -1:
                ##  It is a compulsory miss
                pc_miss_histo[pc] = pc_miss_histo[pc] + count
            else:
                ##  It is a hit
                pc_hit_histo[pc] = pc_hit_histo[pc] + count
        else:
            ##  It is a miss
            pc_miss_histo[pc] = pc_miss_histo[pc] + count

    ##  Call the function to do drawing
    draw_pc_histo(pcs, list(pc_hit_histo.values), list(pc_miss_histo.values), out_file_path)


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
    file_content = file_content.strip()

    ##  Convert file content to a list of tuples(pc, distance, count), for easier later proceeed
    distance_records = list()
    ##  Iterate over each line
    lines = file_content.split('\n')
    for line in lines:
        ##  If an empty line encountered, just skip
        if len(line.strip()) < 1:
            continue
        str_nums = line.split(' ')
        distance_records.append((int(str_nums[0]), int(str_nums[1]), int(str_nums[2])))

    ##  Call the function to do miss type analysis
    miss_type_analyze(distance_records, path.join(out_dir_bench, kernel + "_miss_type.png"))

    ##  Call the function to do distance histo analysis
    distance_histo_analyze(distance_records, path.join(out_dir_bench, kernel + "_distance_histo.png"))

    ##  Call the function to do pc histo analysis
    pc_histo_analyze(distance_records, path.join(out_dir_bench, kernel + "_pc_histo.png"))




if __name__ == '__main__':
    main()
