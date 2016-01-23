#!/usr/bin/python3

import os.path as path


from pandas import Series, DataFrame
import pandas
import numpy as np
import matplotlib.pyplot as plt


from parser import *

def draw_error_comparison(miss_frame, suite):
    ##  Get subframe for this suite
    sub_frame = miss_frame[miss_frame['suite'] == suite]

    kernels = sub_frame['kernel']
    index = np.arange(len(kernels))

    ##  Calculate errors
    error_model = (sub_frame['model_miss'] - sub_frame['profiler_miss']).abs() * 100
    error_base_model = (sub_frame['base_model_miss'] - sub_frame['profiler_miss']).abs() * 100
    error_sim = (sub_frame['sim_miss'] - sub_frame['profiler_miss']).abs() * 100

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
    plt.legend(loc = 'lower left', fontsize = 'small', ncol = 3, bbox_to_anchor = (0, 1))

    ##  Write figure to file and clear plot
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(path.join(dir_script, "../output/miss_rate_error_compare_" + suite + ".png"))
    fig.clf()


#def draw_miss_parboil(miss_frame):
#    ##  Draw miss rate polybench
#    kernel_names = miss_frame['kernel'][0:22]
#    index = np.arange(len(kernel_names))
#
#    rect1 = plt.bar(index + 0.1, miss_frame['profiler_miss'][0:22], 0.2, label = 'profiler', color = 'w', hatch = '*')
#    rect2 = plt.bar(index + 0.3, miss_frame['model_miss'][0:22], 0.2, label = 'model', color = 'w', hatch = '---')
#    rect1 = plt.bar(index + 0.5, miss_frame['base_model_miss'][0:22], 0.2, label = 'base model', color = 'w', hatch = '///')
#    rect1 = plt.bar(index + 0.7, miss_frame['sim_miss'][0:22], 0.2, label = 'GPGPU-Sim', color = 'w', hatch = 'x')
#
#    plt.xlim(0, len(kernel_names))
#    plt.xticks(index + 0.5, kernel_names, rotation = 'vertical')
#    ##plt.title('l1 missrate of profiler and model')
#    plt.title('llllll ')
#    plt.xlabel('kernel name')
#    plt.ylabel('miss rate')
#    plt.legend(loc = 'lower left', fontsize = 'small', ncol = 5, bbox_to_anchor = (0, 1))
#
#    fig = plt.gcf()
#    fig.set_size_inches(14, 8)
#    fig.set_dpi(72)
#    fig.set_tight_layout(True)
#    fig.savefig(path.join(dir_script, "../output/miss_rate_parboil.png"))
#    fig.clf()
#
#def draw_miss_polybench_gpu(miss_frame):
#     ##  Draw miss rate polybench
#    kernel_names = miss_frame['kernel'][22:42]
#    index = np.arange(len(kernel_names))
#    kernel_names.index = index
#
#    rect1 = plt.bar(index + 0.1, miss_frame['profiler_miss'][22:42], 0.2, label = 'profiler', color = 'w', hatch = '*')
#    rect2 = plt.bar(index + 0.3, miss_frame['model_miss'][22:42], 0.2, label = 'model', color = 'w', hatch = '---')
#    rect1 = plt.bar(index + 0.5, miss_frame['base_model_miss'][22:42], 0.2, label = 'base model', color = 'w', hatch = '///')
#    rect1 = plt.bar(index + 0.7, miss_frame['sim_miss'][22:42], 0.2, label = 'GPGPU-Sim', color = 'w', hatch = 'x')
#
#    plt.xlim(0, len(kernel_names))
#    plt.xticks(index + 0.5, kernel_names, rotation = 'vertical')
#    ##plt.title('l1 missrate of profiler and model')
#    plt.title('llllll ')
#    plt.xlabel('kernel name')
#    plt.ylabel('miss rate')
#    plt.legend(loc = 'lower left', fontsize = 'small', ncol = 5, bbox_to_anchor = (0, 1))
#
#    fig = plt.gcf()
#    fig.set_size_inches(14, 8)
#    fig.set_dpi(72)
#    fig.set_tight_layout(True)
#    fig.savefig(path.join(dir_script, "../output/miss_rate_polybench_gpu.png"))
#    fig.clf()

#def draw_duration_parboil(duration_frame):
#    ##  Draw miss rate polybench
#    kernel_names = duration_frame['kernel'][0:22]
#    index = np.arange(len(kernel_names))
#
#    #rect1 = plt.bar(index + 0.05, duration_frame['trace'][0:22] + duration_frame['model'][0:22], 0.3, label = 'model', color = 'w', hatch = '*')
#    rect1 = plt.bar(index + 0.05, duration_frame['model'][0:22], 0.3, label = 'model', color = 'w', hatch = '*')
#    #rect2 = plt.bar(index + 0.35, duration_frame['base_trace'][0:22] + duration_frame['base_model'][0:22], 0.3, label = 'base_model', color = 'w', hatch = '---')
#    rect2 = plt.bar(index + 0.35, duration_frame['base_model'][0:22], 0.3, label = 'base_model', color = 'w', hatch = '---')
#    rect1 = plt.bar(index + 0.65, duration_frame['sim'][0:22], 0.3, label = 'GPGPU-Sim', color = 'w', hatch = 'x')
#
#    plt.xlim(0, len(kernel_names))
#    plt.xticks(index + 0.5, kernel_names, rotation = 'vertical')
#    ##plt.title('l1 missrate of profiler and model')
#    plt.title('llllll ')
#    plt.xlabel('kernel name')
#    plt.ylabel('running time')
#    plt.legend(loc = 'lower left', fontsize = 'small', ncol = 5, bbox_to_anchor = (0, 1))
#
#    fig = plt.gcf()
#    fig.set_size_inches(14, 8)
#    fig.set_dpi(72)
#    fig.set_tight_layout(True)
#    fig.savefig(path.join(dir_script, "../output/duration_parboil.png"))
#    fig.clf()
#
#def draw_duration_polybench_gpu(duration_frame):
#     ##  Draw miss rate polybench
#    kernel_names = duration_frame['kernel'][22:42]
#    index = np.arange(len(kernel_names))
#    kernel_names.index = index
#
#    #rect1 = plt.bar(index + 0.05, duration_frame['trace'][22:42] + duration_frame['model'][22:42], 0.3, label = 'model', color = 'w', hatch = '*')
#    rect1 = plt.bar(index + 0.05, duration_frame['model'][22:42], 0.3, label = 'model', color = 'w', hatch = '*')
#    #rect2 = plt.bar(index + 0.35, duration_frame['base_trace'][22:42] + duration_frame['base_model'][22:42], 0.3, label = 'base_model', color = 'w', hatch = '---')
#    rect2 = plt.bar(index + 0.35, duration_frame['base_model'][22:42], 0.3, label = 'base_model', color = 'w', hatch = '---')
#    rect1 = plt.bar(index + 0.65, duration_frame['sim'][22:42], 0.3, label = 'GPGPU-Sim', color = 'w', hatch = '///')
#
#    plt.xlim(0, len(kernel_names))
#    plt.xticks(index + 0.5, kernel_names, rotation = 'vertical')
#    ##plt.title('l1 missrate of profiler and model')
#    plt.title('llllll ')
#    plt.xlabel('kernel name')
#    plt.ylabel('miss rate')
#    plt.legend(loc = 'lower left', fontsize = 'small', ncol = 5, bbox_to_anchor = (0, 1))
#
#    fig = plt.gcf()
#    fig.set_size_inches(14, 8)
#    fig.set_dpi(72)
#    fig.set_tight_layout(True)
#    fig.savefig(path.join(dir_script, "../output/duration_polybench_gpu.png"))
#    fig.clf()

def draw_duration(duration_frame):
    benches = duration_frame['bench']
    index = np.arange(len(benches))


    ##  Plot lines for each
    print(duration_frame.model)
    plt.plot(index + 0.5, duration_frame['model'], linestyle = "-", color = "k", marker = 'o', label = "My Model")
    plt.plot(index + 0.5, duration_frame['base_model'], linestyle = "-", color = "k", marker = 's', label = "Base Model")
    plt.plot(index + 0.5, duration_frame['sim'], linestyle = "-", color = "k", marker = '^', label = "GPGPU-Sim")

    plt.xlim(0, len(benches))
    plt.xticks(index + 0.5, benches, rotation = 'vertical')
    plt.title('Time Consumption Comparison')
    plt.xlabel('Bench Names')
    plt.ylabel('Time Used (ms)')
    plt.legend(loc = 'lower left', fontsize = 'small', ncol = 5, bbox_to_anchor = (0, 1))

    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(path.join(dir_script, "../output/duration.png"))
    fig.clf()




def main():
    ##  Read miss rate and duration DataFrame from csv file
    miss_frame = pandas.read_csv(path.join(dir_script, "../output/miss_rate.csv"))
    duration_frame = pandas.read_csv(path.join(dir_script, "../output/duration.csv"))

    ##  Get all suite names
    ##  Draw a Error comparison chart for each suite
    suites = list(set(miss_frame['suite'].values))
    suites.sort()
    for suite in suites:
        draw_error_comparison(miss_frame, suite)

    #draw_miss_parboil(miss_frame)
    #draw_miss_polybench_gpu(miss_frame)

    ##  Divide duration time by profiler duration for the same bench
    duration_frame['base_model'] = duration_frame['base_model'] / duration_frame['profiler']
    duration_frame['model'] = duration_frame['model'] / duration_frame['profiler']
    duration_frame['base_trace'] = duration_frame['base_trace'] / duration_frame['profiler']
    duration_frame['trace'] = duration_frame['trace'] / duration_frame['profiler']
    duration_frame['sim'] = duration_frame['sim'] / duration_frame['profiler']

    ##  Call function to draw duration comparison
    #draw_duration(duration_frame)

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
