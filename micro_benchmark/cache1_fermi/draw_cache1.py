#!/usr/bin/python3

import csv
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas

#def get_fields_from_csv():
#    file_handle = open('cache1.csv', 'r')
#    csv_reader = csv.reader(file_handle)
#    fields = list()
#    fields.append(list())
#    fields.append(list())
#    fields.append(list())
#    is_header_line = True
#
#    for line in csv_reader:
#        if is_header_line:
#            is_header_line = False
#            continue
#        fields[0].append(int(line[0]))
#        fields[1].append(int(line[1]))
#        fields[2].append(float(line[2]))
#    file_handle.close()
#    return fields

def main():
#    fields = get_fields_from_csv()
#    strides = list(set(fields[0]))
#    strides.sort()
#    num_array_sizes = len(fields[0]) // len(strides)
#
#    array_size = fields[1][0 : num_array_sizes]
#    array_size = [8 * i for i in array_size]
#
#    stride_id = 0
#    for stride in strides:
#        miss_rate = fields[2][stride_id * num_array_sizes : (stride_id + 1) * num_array_sizes]
#
#        ax = plt.subplot(len(strides) // 2, 2, stride_id + 1)
#        ax.set_title("Stride "  + str(stride * 4))
#        ax.set_xlabel("array size")
#        ax.set_ylabel('l1 miss rate(%)')
#        ax.plot(array_size, miss_rate)
#        ax.grid()
#
#        stride_id = stride_id + 1
#
#    fig = plt.gcf()
#    fig.set_size_inches(12, 20)
#    fig.set_dpi(72)
#    fig.set_tight_layout(True)
#    fig.savefig("cache1.png")

    miss_frame = pandas.read_csv("cache1.csv")
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
        ax.grid()

    fig = plt.gcf()
    fig.set_size_inches(12, 10 * len(strides))
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig("cache1.png")

if __name__ == '__main__':
    main()
