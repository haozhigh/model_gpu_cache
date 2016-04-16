#!/usr/bin/python3

import os.path as path


from pandas import Series, DataFrame
import pandas
import numpy as np
import matplotlib.pyplot as plt


from drawer import *

def main():
    ##  Read miss rate DataFrame from csv file
    maxwell_miss_frame = pandas.read_csv(path.join(dir_script, "../output/maxwell_miss_rate.csv"))


    ##  Calculate even errors and print out
    maxwell_model_miss_error = maxwell_miss_frame['model_miss'] - maxwell_miss_frame['profiler_miss']
    maxwell_model_miss_error = maxwell_model_miss_error.abs()

    print("##  Even erros for maxwell model  ##")
    print("maxwell model miss error mean : " + str(maxwell_model_miss_error.mean()))
    print("")

    ##  Write even results to a csv file
    maxwell_even_error_index = [0]
    maxwell_even_error_frame = DataFrame(index = maxwell_even_error_index)
    maxwell_even_error_frame['model'] = maxwell_model_miss_error.mean()

    ##  Write to file
    maxwell_even_error_out_file = path.join(dir_script, "../output/maxwell_even_error.csv")
    maxwell_even_error_frame.to_csv(maxwell_even_error_out_file)

if __name__ == '__main__':
    main()
