#!/usr/bin/python3

import os.path as path


from pandas import Series, DataFrame
import pandas
import numpy as np
import matplotlib.pyplot as plt


from drawer import *

def main():
    ##  Read even error DataFrame from csv file
    even_error_frame = pandas.read_csv(path.join(dir_script, "../output/even_error.csv"))

    ##  Do the drawing
    draw_even_error(even_error_frame, "Even Error Comparison", path.join(dir_script, "../output/even_error.png"))


if __name__ == '__main__':
    main()
