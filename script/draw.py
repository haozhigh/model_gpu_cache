#!/usr/bin/python3

import os.path as path


from pandas import Series, DataFrame
import pandas
import numpy as np
import matplotlib.pyplot as plt


from common import *


def draw_miss_parboil(miss_frame):
    ##  Draw miss rate polybench
    kernel_names = miss_frame['kernel'][0:22]
    index = np.arange(len(kernel_names))

    rect1 = plt.bar(index + 0.1, miss_frame['profiler_miss'][0:22], 0.2, label = 'profiler', color = 'w', hatch = '*')
    rect2 = plt.bar(index + 0.3, miss_frame['model_miss'][0:22], 0.2, label = 'model', color = 'w', hatch = '---')
    rect1 = plt.bar(index + 0.5, miss_frame['base_model_miss'][0:22], 0.2, label = 'base model', color = 'w', hatch = '///')
    rect1 = plt.bar(index + 0.7, miss_frame['sim_miss'][0:22], 0.2, label = 'GPGPU-Sim', color = 'w', hatch = 'x')

    plt.xlim(0, len(kernel_names))
    plt.xticks(index + 0.5, kernel_names, rotation = 'vertical')
    ##plt.title('l1 missrate of profiler and model')
    plt.title('llllll ')
    plt.xlabel('kernel name')
    plt.ylabel('miss rate')
    plt.legend(loc = 'lower left', fontsize = 'small', ncol = 5, bbox_to_anchor = (0, 1))

    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(path.join(dir_script, "../output/miss_rate_parboil.png"))
    fig.clf()

def draw_miss_polybench_gpu(miss_frame):
     ##  Draw miss rate polybench
    kernel_names = miss_frame['kernel'][22:42]
    index = np.arange(len(kernel_names))
    kernel_names.index = index

    rect1 = plt.bar(index + 0.1, miss_frame['profiler_miss'][22:42], 0.2, label = 'profiler', color = 'w', hatch = '*')
    rect2 = plt.bar(index + 0.3, miss_frame['model_miss'][22:42], 0.2, label = 'model', color = 'w', hatch = '---')
    rect1 = plt.bar(index + 0.5, miss_frame['base_model_miss'][22:42], 0.2, label = 'base model', color = 'w', hatch = '///')
    rect1 = plt.bar(index + 0.7, miss_frame['sim_miss'][22:42], 0.2, label = 'GPGPU-Sim', color = 'w', hatch = 'x')

    plt.xlim(0, len(kernel_names))
    plt.xticks(index + 0.5, kernel_names, rotation = 'vertical')
    ##plt.title('l1 missrate of profiler and model')
    plt.title('llllll ')
    plt.xlabel('kernel name')
    plt.ylabel('miss rate')
    plt.legend(loc = 'lower left', fontsize = 'small', ncol = 5, bbox_to_anchor = (0, 1))

    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(path.join(dir_script, "../output/miss_rate_polybench_gpu.png"))
    fig.clf()

def draw_duration_parboil(duration_frame):
    ##  Draw miss rate polybench
    kernel_names = duration_frame['kernel'][0:22]
    index = np.arange(len(kernel_names))

    #rect1 = plt.bar(index + 0.05, duration_frame['trace'][0:22] + duration_frame['model'][0:22], 0.3, label = 'model', color = 'w', hatch = '*')
    rect1 = plt.bar(index + 0.05, duration_frame['model'][0:22], 0.3, label = 'model', color = 'w', hatch = '*')
    #rect2 = plt.bar(index + 0.35, duration_frame['base_trace'][0:22] + duration_frame['base_model'][0:22], 0.3, label = 'base_model', color = 'w', hatch = '---')
    rect2 = plt.bar(index + 0.35, duration_frame['base_model'][0:22], 0.3, label = 'base_model', color = 'w', hatch = '---')
    rect1 = plt.bar(index + 0.65, duration_frame['sim'][0:22], 0.3, label = 'GPGPU-Sim', color = 'w', hatch = 'x')

    plt.xlim(0, len(kernel_names))
    plt.xticks(index + 0.5, kernel_names, rotation = 'vertical')
    ##plt.title('l1 missrate of profiler and model')
    plt.title('llllll ')
    plt.xlabel('kernel name')
    plt.ylabel('running time')
    plt.legend(loc = 'lower left', fontsize = 'small', ncol = 5, bbox_to_anchor = (0, 1))

    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(path.join(dir_script, "../output/duration_parboil.png"))
    fig.clf()

def draw_duration_polybench_gpu(duration_frame):
     ##  Draw miss rate polybench
    kernel_names = duration_frame['kernel'][22:42]
    index = np.arange(len(kernel_names))
    kernel_names.index = index

    #rect1 = plt.bar(index + 0.05, duration_frame['trace'][22:42] + duration_frame['model'][22:42], 0.3, label = 'model', color = 'w', hatch = '*')
    rect1 = plt.bar(index + 0.05, duration_frame['model'][22:42], 0.3, label = 'model', color = 'w', hatch = '*')
    #rect2 = plt.bar(index + 0.35, duration_frame['base_trace'][22:42] + duration_frame['base_model'][22:42], 0.3, label = 'base_model', color = 'w', hatch = '---')
    rect2 = plt.bar(index + 0.35, duration_frame['base_model'][22:42], 0.3, label = 'base_model', color = 'w', hatch = '---')
    rect1 = plt.bar(index + 0.65, duration_frame['sim'][22:42], 0.3, label = 'GPGPU-Sim', color = 'w', hatch = '///')

    plt.xlim(0, len(kernel_names))
    plt.xticks(index + 0.5, kernel_names, rotation = 'vertical')
    ##plt.title('l1 missrate of profiler and model')
    plt.title('llllll ')
    plt.xlabel('kernel name')
    plt.ylabel('miss rate')
    plt.legend(loc = 'lower left', fontsize = 'small', ncol = 5, bbox_to_anchor = (0, 1))

    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(path.join(dir_script, "../output/duration_polybench_gpu.png"))
    fig.clf()




def main():
    miss_frame = pandas.read_csv(path.join(dir_script, "../output/miss_rate.csv"))
    duration_frame = pandas.read_csv(path.join(dir_script, "../output/duration.csv"))


    draw_miss_parboil(miss_frame)
    draw_miss_polybench_gpu(miss_frame)

    duration_frame['base_model'] = duration_frame['base_model'] / duration_frame['profiler']
    duration_frame['model'] = duration_frame['model'] / duration_frame['profiler']
    duration_frame['base_trace'] = duration_frame['base_trace'] / duration_frame['profiler']
    duration_frame['trace'] = duration_frame['trace'] / duration_frame['profiler']
    duration_frame['sim'] = duration_frame['sim'] / duration_frame['profiler']

    for wide_kernel_name in duration_frame.index:
        if duration_frame['sim'][wide_kernel_name] < 0:
            duration_frame['sim'][wide_kernel_name] = duration_frame['trace'][wide_kernel_name] + duration_frame['model'][wide_kernel_name]

    draw_duration_parboil(duration_frame)
    draw_duration_polybench_gpu(duration_frame)

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
