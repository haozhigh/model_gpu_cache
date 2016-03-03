#!/usr/bin/python3

import os.path as path


from pandas import Series, DataFrame
import pandas
import numpy as np
import matplotlib.pyplot as plt


from parser import *

def draw_error_comparison(miss_frame, title, save_path):
    kernels = miss_frame['kernel']
    index = np.arange(len(kernels))

    ##  Calculate errors
    error_model = (miss_frame['model_miss'] - miss_frame['profiler_miss']).abs() * 100
    error_base_model = (miss_frame['base_model_miss'] - miss_frame['profiler_miss']).abs() * 100
    error_sim = (miss_frame['sim_miss'] - miss_frame['profiler_miss']).abs() * 100

    ##  Do the plot
    plt.plot(index + 0.5, error_model, linestyle = "-", color = "k", marker = 'o', label = "My Model")
    plt.plot(index + 0.5, error_base_model, linestyle = "-", color = "k", marker = 's', label = "Base Model")
    plt.plot(index + 0.5, error_sim, linestyle = "-", color = "k", marker = '^', label = "GPGPU-Sim")

    ##  Figure options set
    plt.xlim(0, len(kernels))
    plt.xticks(index + 0.5, kernels.values, rotation = 'vertical')
    plt.title(title)
    plt.xlabel('Kernel Name')
    plt.ylabel('Miss Rate Error(%)')
    plt.legend()

    ##  Write figure to file and clear plot
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(save_path)
    fig.clf()

def draw_architecture_compare(fermi_miss_frame, maxwell_miss_frame, title, save_path):
    kernels = fermi_miss_frame['kernel']
    index = np.arange(len(kernels))

    ##  Do the plot
    plt.plot(index + 0.5, fermi_miss_frame['profiler_miss'], linestyle = "-", color = "k", marker = 'o', label = "Fermi Profiler Miss Rate")
    plt.plot(index + 0.5, maxwell_miss_frame['profiler_miss'], linestyle = "-", color = "k", marker = 's', label = "MaxWell Profiler Miss Rate")

    ##  Figure options set
    plt.xlim(0, len(kernels))
    plt.xticks(index + 0.5, kernels.values, rotation = 'vertical')
    plt.title(title)
    plt.xlabel('Kernel Name')
    plt.ylabel('Miss Rate Error(%)')
    plt.legend()

    ##  Write figure to file and clear plot
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(save_path)
    fig.clf()


def draw_error_model(miss_frame, title, save_path):
    kernels = miss_frame['kernel']
    index = np.arange(len(kernels))

    ##  Calculate errors
    error_model = (miss_frame['model_miss'] - miss_frame['profiler_miss']).abs() * 100

    ##  Do the plot
    plt.plot(index + 0.5, error_model, linestyle = "-", color = "k", marker = 'o', label = "My Model")

    ##  Figure options set
    plt.xlim(0, len(kernels))
    plt.xticks(index + 0.5, kernels.values, rotation = 'vertical')
    plt.title(title)
    plt.xlabel('Kernel Name')
    plt.ylabel('Miss Rate Error(%)')
    plt.legend()

    ##  Write figure to file and clear plot
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(save_path)
    fig.clf()


def draw_duration(duration_frame, title, save_path):
    benches = duration_frame['bench']
    index = np.arange(len(benches))


    ##  Plot lines for each
    plt.plot(index + 0.5, duration_frame['model'], linestyle = "-", color = "k", marker = 'o', label = "My Model")
    plt.plot(index + 0.5, duration_frame['base_model'], linestyle = "-", color = "k", marker = 's', label = "Base Model")
    plt.plot(index + 0.5, duration_frame['sim'], linestyle = "-", color = "k", marker = '^', label = "GPGPU-Sim")

    plt.xlim(0, len(benches))
    plt.xticks(index + 0.5, benches, rotation = 'vertical')
    plt.title(title)
    plt.xlabel('Bench Names')
    plt.ylabel('Time Used (ms)')
    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(save_path)
    fig.clf()


def draw_footprint(footprint_frame, title, save_path):
    kernels = footprint_frame['kernel']
    index = np.arange(len(kernels))

    plt.plot(index + 0.5, footprint_frame['footprint'] / 1024, linestyle = "-", color = "k", marker = "o", label = "Memory Footprint")

    plt.xlim(0, len(kernels))
    plt.xticks(index + 0.5, kernels, rotation = 'vertical')
    plt.title(title)
    plt.xlabel('Kernel Names')
    plt.ylabel('Memory Footprint (KB)')
    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(save_path)
    fig.clf()


def draw_op_intensity(footprint_frame, title, save_path):
    kernels = footprint_frame['kernel']
    index = np.arange(len(kernels))

    plt.plot(index + 0.5, footprint_frame['op_intensity'], linestyle = "-", color = "k", marker = "o", label = "Operation Intensity")

    plt.xlim(0, len(kernels))
    plt.xticks(index + 0.5, kernels, rotation = 'vertical')
    plt.title(title)
    plt.xlabel('Kernel Names')
    plt.ylabel('Operation Intensity (1)')
    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(save_path)
    fig.clf()

def draw_miss_rate(footprint_frame, title, save_path):
    kernels = footprint_frame['kernel']
    index = np.arange(len(kernels))

    plt.plot(index + 0.5, footprint_frame['profiler_miss'] * 100, linestyle = "-", color = "k", marker = "o", label = "Miss Rate")

    plt.xlim(0, len(kernels))
    plt.xticks(index + 0.5, kernels, rotation = 'vertical')
    plt.title(title)
    plt.xlabel('Kernel Names')
    plt.ylabel('Miss Rate (%)')
    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(save_path)
    fig.clf()

def draw_distance_histo(distance_values, distance_counts, save_path):
    index = np.arange(len(distance_values))

    rects = plt.bar(index + 0.1, distance_counts, 0.8, color = 'w')

#    for i in range(0, len(rects)):
#        rect = rects[i]
#        plt.text(rect.get_x() + rect.get_width() / 2, rect.get_height() * 1.01, '%d'%(distance_counts[i]), ha = 'center', va = 'baseline')

    plt.xlim(0, len(distance_values))
    plt.ylim(0, max(distance_counts) * 1.2)
    #plt.xticks(index + 0.5, distance_values, rotation = 'vertical')
    plt.xticks(index + 0.5, distance_values)
    plt.xlabel('Reuse Distance')
    plt.ylabel('Occurance Count')

    fig = plt.gcf()
    fig.set_size_inches(6, 3)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(save_path)
    fig.clf()

def draw_pc_histo(pcs, pc_hit_counts, pc_miss_counts, save_path):
    index = np.arange(len(pcs))

    plt.bar(index + 0.1, pc_miss_counts, 0.4, color = 'w', label = "Miss Count", hatch = "//")
    plt.bar(index + 0.5, pc_hit_counts, 0.4, color = 'w', label = "Hit Count", hatch = "..")

    plt.xlim(0, len(pcs))
    plt.ylim(0, max(max(pc_miss_counts), max(pc_hit_counts)) * 1.2)
    #plt.xticks(index + 0.5, distance_values, rotation = 'vertical')
    plt.xticks(index + 0.5, pcs)
    plt.title("-")
    plt.xlabel('Program Counter')
    plt.ylabel('Cache Miss/Hit Count')
    plt.legend(loc = 'lower left', fontsize = 'small', ncol = 2, bbox_to_anchor = (0, 1))

    fig = plt.gcf()
    fig.set_size_inches(6, 3)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(save_path)
    fig.clf()

def draw_miss_breakdown(miss_frame, save_path):
    kernels = miss_frame['kernel']
    index = np.arange(len(kernels))

    plt.bar(index + 0.15, miss_frame['model_comp_miss'], 0.35, label = 'compulsory miss', color = 'w', hatch = '//')
    plt.bar(index + 0.50, miss_frame['model_uncomp_miss'], 0.35, label = 'uncompulsory miss', color = 'w', hatch = '..', bottom = miss_frame['model_comp_miss'])

    ##  Figure options set
    plt.xlim(0, len(kernels))
    plt.xticks(index + 0.5, kernels.values, rotation = 'vertical')
    plt.title('-')
    plt.xlabel('Kernel Name')
    plt.ylabel('Miss Rate(%)')
    plt.legend(loc = 'lower left', fontsize = 'small', ncol = 2, bbox_to_anchor = (0, 1))

    ##  Write figure to file and clear plot
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(save_path)
    fig.clf()



def main():
    print("This is an empty main function.")



if __name__ == '__main__':
    main()
