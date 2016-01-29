#!/usr/bin/python3

import csv
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas

def main():
    miss_frame = pandas.read_csv("result.csv")


    strides = list(set(miss_frame['stride']))
    strides.sort()

    stride_id = 1
    for stride in strides:
        sub_frame = miss_frame[miss_frame['stride'] == stride]

        ax = plt.subplot(len(strides), 1, stride_id)
        stride_id = stride_id + 1

        ax.plot(sub_frame['array_size'] * 8, sub_frame['l1_miss_rate'])

        ax.set_title("Stride " + str(stride * 8) + " Bytes")
        ax.set_xlabel("Array Size (Bytes)")
        ax.set_ylabel("L1 Cache Miss Rate (%)")
        ax.set_ylim(0, sub_frame['l1_miss_rate'].max() + 5)
        ax.grid()

    fig = plt.gcf()
    fig.set_size_inches(12, 10 * len(strides))
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig("result.png")

if __name__ == '__main__':
    main()
