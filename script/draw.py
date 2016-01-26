#!/usr/bin/python3

import os.path as path


from pandas import Series, DataFrame
import pandas
import numpy as np
import matplotlib.pyplot as plt


from parser import *

def draw_error_comparison(miss_frame):
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
    plt.title('Miss Rate Error Comparison')
    plt.xlabel('Kernel Name')
    plt.ylabel('Miss Rate Error(%)')
    plt.legend()

    ##  Write figure to file and clear plot
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(path.join(dir_script, "../output/miss_rate_error_compare.png"))
    fig.clf()


def draw_duration(duration_frame):
    benches = duration_frame['bench']
    index = np.arange(len(benches))


    ##  Plot lines for each
    plt.plot(index + 0.5, duration_frame['model'], linestyle = "-", color = "k", marker = 'o', label = "My Model")
    plt.plot(index + 0.5, duration_frame['base_model'], linestyle = "-", color = "k", marker = 's', label = "Base Model")
    plt.plot(index + 0.5, duration_frame['sim'], linestyle = "-", color = "k", marker = '^', label = "GPGPU-Sim")

    plt.xlim(0, len(benches))
    plt.xticks(index + 0.5, benches, rotation = 'vertical')
    plt.title('Time Consumption Comparison')
    plt.xlabel('Bench Names')
    plt.ylabel('Time Used (ms)')
    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(path.join(dir_script, "../output/duration.png"))
    fig.clf()


def draw_footprint(footprint_frame):
    kernels = footprint_frame['kernel']
    index = np.arange(len(kernels))

    plt.plot(index + 0.5, footprint_frame['footprint'] / 1024, linestyle = "-", color = "k", marker = "o", label = "Memory Footprint")

    plt.xlim(0, len(kernels))
    plt.xticks(index + 0.5, kernels, rotation = 'vertical')
    plt.title('Kernel Memory Footprint')
    plt.xlabel('Kernel Names')
    plt.ylabel('Memory Footprint (KB)')
    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(path.join(dir_script, "../output/footprint.png"))
    fig.clf()


def draw_op_intensity(footprint_frame):
    kernels = footprint_frame['kernel']
    index = np.arange(len(kernels))

    plt.plot(index + 0.5, footprint_frame['op_intensity'], linestyle = "-", color = "k", marker = "o", label = "Operation Intensity")

    plt.xlim(0, len(kernels))
    plt.xticks(index + 0.5, kernels, rotation = 'vertical')
    plt.title('Operation intensity for each kernel')
    plt.xlabel('Kernel Names')
    plt.ylabel('Operation Intensity (1)')
    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(path.join(dir_script, "../output/op_intensity.png"))
    fig.clf()

def draw_miss_rate(footprint_frame):
    kernels = footprint_frame['kernel']
    index = np.arange(len(kernels))

    plt.plot(index + 0.5, footprint_frame['profiler_miss'] * 100, linestyle = "-", color = "k", marker = "o", label = "Miss Rate")

    plt.xlim(0, len(kernels))
    plt.xticks(index + 0.5, kernels, rotation = 'vertical')
    plt.title('L1 cache miss rate for each kernel')
    plt.xlabel('Kernel Names')
    plt.ylabel('Miss Rate (%)')
    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(path.join(dir_script, "../output/miss_rate.png"))
    fig.clf()



def main():
    ##  Read miss rate DataFrame from csv file
    miss_frame = pandas.read_csv(path.join(dir_script, "../output/miss_rate.csv"))

    ##  Get all suite names
    ##  Draw a Error comparison chart for each suite
    draw_error_comparison(miss_frame)

    ##  Divide duration time by profiler duration for the same bench
    #duration_frame['base_model'] = duration_frame['base_model'] / duration_frame['profiler']
    #duration_frame['model'] = duration_frame['model'] / duration_frame['profiler']
    #duration_frame['base_trace'] = duration_frame['base_trace'] / duration_frame['profiler']
    #duration_frame['trace'] = duration_frame['trace'] / duration_frame['profiler']
    #duration_frame['sim'] = duration_frame['sim'] / duration_frame['profiler']





    ##  Read duration time from csv file
    duration_frame = pandas.read_csv(path.join(dir_script, "../output/duration.csv"))

    ##  Add trace generation time to total model time
    duration_frame['base_model'] = duration_frame['base_model'] + duration_frame['base_trace']
    duration_frame['model'] = duration_frame['model'] + duration_frame['trace']

    ##  Call function to draw duration comparison
    draw_duration(duration_frame)





    ##  Read memory footprint info fram csv file
    footprint_frame = pandas.read_csv(path.join(dir_script, "../output/footprint.csv"))

    ##  Draw memory footprint
    draw_footprint(footprint_frame)

    ##  Draw operation intensity
    draw_op_intensity(footprint_frame)

    ##  Draw miss rate
    draw_miss_rate(footprint_frame)




    ##  Calculate even values
    model_miss_error = miss_frame['model_miss'] - miss_frame['profiler_miss']
    model_miss_error = model_miss_error.abs()
    base_model_miss_error = miss_frame['base_model_miss'] - miss_frame['profiler_miss']
    base_model_miss_error = base_model_miss_error.abs()
    sim_miss_error = miss_frame['sim_miss'] - miss_frame['profiler_miss']
    sim_miss_error = sim_miss_error.abs()

    print("model miss error mean : " + str(model_miss_error.mean()))
    print("base_model miss error mean : " + str(base_model_miss_error.mean()))
    print("sim miss error mean : " + str(sim_miss_error.mean()))

    model_duration = duration_frame['model'] + duration_frame['trace']
    base_model_duration = duration_frame['base_model'] + duration_frame['base_trace']
    sim_duration = duration_frame['sim']

    print("model duration mean: " + str(model_duration.mean()))
    print("base model duration mean: " + str(base_model_duration.mean()))
    print("sim duration mean: " + str(sim_duration.mean()))

    ##  Exclude trace generation time
    model_duration = duration_frame['model']
    base_model_duration = duration_frame['base_model']
    sim_duration = duration_frame['sim']

    print("model duration mean: " + str(model_duration.mean()))
    print("base model duration mean: " + str(base_model_duration.mean()))
    print("sim duration mean: " + str(sim_duration.mean()))



if __name__ == '__main__':
    main()
