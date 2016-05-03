#!/usr/bin/python3

import os.path as path


from pandas import Series, DataFrame
import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


from parser import *


##  cn_font is shared among all function in this file
cn_font = FontProperties(fname = r"./fonts/simsun.ttc", size = 10)

def get_bench_index(frame):
    benches = frame['bench'].values
    benches.sort()

    for i in range(0, len(benches) - 1):
        j = i + 1
        while ((j < len(benches)) and (benches[j] == benches[i])):
            benches[j] = benches[j] + str(j - i)
            j = j + 1
        if (j > i + 1):
            benches[i] = benches[i] + str(0)

    return benches

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

def draw_error_comparison_v2(miss_frame, title, save_path):
    suites = list(set(miss_frame['suite'].values))
    suites.sort()

    num_suites = len(suites)
    suite_id = 0
    for suite in suites:
        suite_id = suite_id + 1
        ax = plt.subplot(num_suites, 1, suite_id)
        sub_frame = miss_frame.ix[miss_frame['suite'] == suite]

        ##  Calculate errors
        error_model = (sub_frame['model_miss'] - sub_frame['profiler_miss']).abs() * 100
        error_base_model = (sub_frame['base_model_miss'] - sub_frame['profiler_miss']).abs() * 100
        error_sim = (sub_frame['sim_miss'] - sub_frame['profiler_miss']).abs() * 100

        bench_index = get_bench_index(sub_frame)
        index = np.arange(len(bench_index))

        ##  Do the plot
        ax.plot(index + 0.5, error_model, linestyle = "-", color = "k", marker = 'o', label = "本文模型")
        ax.plot(index + 0.5, error_base_model, linestyle = "-", color = "k", marker = 's', label = "Nugteren模型")
        ax.plot(index + 0.5, error_sim, linestyle = "-", color = "k", marker = '^', label = "GPGPU-Sim")

        ax.tick_params(axis='both', which='both', labelsize=10)
        ax.set_xlim(0, len(bench_index))
        ax.set_xticks(index + 0.5)
        ax.set_xticklabels(bench_index, rotation = 45)
        ##ax.set_xlabel('Kernel_nmaes', fontproperties = cn_font)
        ax.set_ylabel("缓存缺失率误差（%）", fontproperties = cn_font)
        ax.set_title(suite + "测试集", fontproperties = cn_font)

        if suite_id == num_suites:
            ax.legend(loc = 'upper left', fontsize = 'small', ncol = 3, bbox_to_anchor = (0, -0.45), prop = cn_font)

    fig = plt.gcf()
    fig.set_size_inches(6, 4)
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

def draw_architecture_compare_v2(fermi_miss_frame, maxwell_miss_frame, title, save_path):
    suites = list(set(fermi_miss_frame['suite'].values))
    suites.sort()

    num_suites = len(suites)
    suite_id = 0
    for suite in suites:
        suite_id = suite_id + 1
        ax = plt.subplot(num_suites, 1, suite_id)
        fermi_sub_frame = fermi_miss_frame.ix[fermi_miss_frame['suite'] == suite]
        maxwell_sub_frame = maxwell_miss_frame.ix[maxwell_miss_frame['suite'] == suite]

        bench_index = get_bench_index(fermi_sub_frame)
        index = np.arange(len(bench_index))

        ##  Do the plot
        ax.plot(index + 0.5, fermi_sub_frame['profiler_miss'], linestyle = "-", color = "k", marker = 'o', label = "费米架构")
        ax.plot(index + 0.5, maxwell_sub_frame['profiler_miss'], linestyle = "-", color = "k", marker = 's', label = "麦克斯韦架构")
        
        ax.tick_params(axis='both', which='both', labelsize=10)
        ax.set_xlim(0, len(bench_index))
        ax.set_xticks(index + 0.5)
        ax.set_xticklabels(bench_index, rotation = 45)
        ##ax.set_xlabel('Kernel_nmaes', fontproperties = cn_font)
        ax.set_ylabel("缓存缺失率（%）", fontproperties = cn_font)
        ax.set_title(suite + "测试集", fontproperties = cn_font)

        if suite_id == num_suites:
            ax.legend(loc = 'upper left', fontsize = 'small', ncol = 2, bbox_to_anchor = (0, -0.45), prop = cn_font)

    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(save_path)
    fig.clf()

def draw_architecture_model_compare_v2(fermi_miss_frame, maxwell_miss_frame, title, save_path):
    suites = list(set(fermi_miss_frame['suite'].values))
    suites.sort()

    num_suites = len(suites)
    suite_id = 0
    for suite in suites:
        suite_id = suite_id + 1
        ax = plt.subplot(num_suites, 1, suite_id)
        fermi_sub_frame = fermi_miss_frame.ix[fermi_miss_frame['suite'] == suite]
        maxwell_sub_frame = maxwell_miss_frame.ix[maxwell_miss_frame['suite'] == suite]

        bench_index = get_bench_index(fermi_sub_frame)
        index = np.arange(len(bench_index))

        ##  Do the plot
        ax.plot(index + 0.5, fermi_sub_frame['model_miss'], linestyle = "-", color = "k", marker = 'o', label = "费米架构")
        ax.plot(index + 0.5, maxwell_sub_frame['model_miss'], linestyle = "-", color = "k", marker = 's', label = "麦克斯韦架构")
        
        ax.tick_params(axis='both', which='both', labelsize=10)
        ax.set_xlim(0, len(bench_index))
        ax.set_xticks(index + 0.5)
        ax.set_xticklabels(bench_index, rotation = 45)
        ##ax.set_xlabel('Kernel_nmaes', fontproperties = cn_font)
        ax.set_ylabel("模型缓存缺失率（%）", fontproperties = cn_font)
        ax.set_title(suite + "测试集", fontproperties = cn_font)

        if suite_id == num_suites:
            ax.legend(loc = 'upper left', fontsize = 'small', ncol = 2, bbox_to_anchor = (0, -0.45), prop = cn_font)

    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(save_path)
    fig.clf()

def draw_architecture_error_compare_v2(fermi_miss_frame, maxwell_miss_frame, title, save_path):
    suites = list(set(fermi_miss_frame['suite'].values))
    suites.sort()

    num_suites = len(suites)
    suite_id = 0
    for suite in suites:
        suite_id = suite_id + 1
        ax = plt.subplot(num_suites, 1, suite_id)
        fermi_sub_frame = fermi_miss_frame.ix[fermi_miss_frame['suite'] == suite]
        maxwell_sub_frame = maxwell_miss_frame.ix[maxwell_miss_frame['suite'] == suite]

        bench_index = get_bench_index(fermi_sub_frame)
        index = np.arange(len(bench_index))

        ##  Calculate errors
        fermi_error_model = (fermi_sub_frame['model_miss'] - fermi_sub_frame['profiler_miss']).abs() * 100
        maxwell_error_model = (maxwell_sub_frame['model_miss'] - maxwell_sub_frame['profiler_miss']).abs() * 100

        ##  Do the plot
        ax.plot(index + 0.5, fermi_error_model, linestyle = "-", color = "k", marker = 'o', label = "费米架构")
        ax.plot(index + 0.5, maxwell_error_model, linestyle = "-", color = "k", marker = 's', label = "麦克斯韦架构")
        
        ax.tick_params(axis='both', which='both', labelsize=10)
        ax.set_xlim(0, len(bench_index))
        ax.set_xticks(index + 0.5)
        ax.set_xticklabels(bench_index, rotation = 45)
        ##ax.set_xlabel('Kernel_nmaes', fontproperties = cn_font)
        ax.set_ylabel("缓存缺失率误差（%）", fontproperties = cn_font)
        ax.set_title(suite + "测试集", fontproperties = cn_font)

        if suite_id == num_suites:
            ax.legend(loc = 'upper left', fontsize = 'small', ncol = 2, bbox_to_anchor = (0, -0.45), prop = cn_font)

    fig = plt.gcf()
    fig.set_size_inches(6, 4)
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

def draw_error_model_v2(miss_frame, title, save_path):
    suites = list(set(miss_frame['suite'].values))
    suites.sort()

    num_suites = len(suites)
    suite_id = 0
    for suite in suites:
        suite_id = suite_id + 1
        ax = plt.subplot(num_suites, 1, suite_id)
        sub_frame = miss_frame.ix[miss_frame['suite'] == suite]

        bench_index = get_bench_index(sub_frame)
        index = np.arange(len(bench_index))

        ##  Calculate errors
        error_model = (sub_frame['model_miss'] - sub_frame['profiler_miss']).abs() * 100

        ##  Do the plot
        ax.plot(index + 0.5, error_model, linestyle = "-", color = "k", marker = 'o', label = "My Model")

        ax.tick_params(axis='both', which='both', labelsize=10)
        ax.set_xlim(0, len(bench_index))
        ax.set_xticks(index + 0.5)
        ax.set_xticklabels(bench_index, rotation = 45)
        ##ax.set_xlabel('Kernel_nmaes', fontproperties = cn_font)
        ax.set_ylabel("模型缓存缺失率误差（%）", fontproperties = cn_font)
        ax.set_title(suite + "测试集", fontproperties = cn_font)

    fig = plt.gcf()
    fig.set_size_inches(6, 4)
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
    #plt.plot(index + 0.5, duration_frame['sim'], linestyle = "-", color = "k", marker = '^', label = "GPGPU-Sim")

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

def draw_duration_v2(duration_frame, title, save_path):
    benches = duration_frame['bench']
    index = np.arange(len(benches))


    ##  Plot lines for each
    plt.plot(index + 0.5, duration_frame['model'], linestyle = "-", color = "k", marker = 'o', label = "本文模型")
    plt.plot(index + 0.5, duration_frame['base_model'], linestyle = "-", color = "k", marker = 's', label = "Nugteren模型")

    plt.tick_params(axis='both', which='both', labelsize=10)
    plt.xlim(0, len(benches))
    plt.xticks(index + 0.5, benches, rotation = 45)
    plt.title("重用距离计算时间开销比较", fontproperties = cn_font)
    plt.ylabel("时间开销（毫秒)", fontproperties = cn_font)
    plt.legend(loc = 'upper left', fontsize = 'small', ncol = 1, bbox_to_anchor = (0, 0.98), prop = cn_font)

    fig = plt.gcf()
    fig.set_size_inches(6, 3)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(save_path)
    fig.clf()

def draw_reuse_distance_duration_comparison_v2(duration_frame, title, save_path):
    suites = list(set(duration_frame['suite'].values))
    suites.sort()

    num_suites = len(suites)
    suite_id = 0
    for suite in suites:
        suite_id = suite_id + 1
        ax = plt.subplot(num_suites, 1, suite_id)
        sub_frame = duration_frame.ix[duration_frame['suite'] == suite]

        bench_index = get_bench_index(sub_frame)
        index = np.arange(len(bench_index))

        ##  Plot lines for each
        plt.plot(index + 0.5, sub_frame['model_compare'], linestyle = "-", color = "k", marker = 'o', label = "本文模型")
        plt.plot(index + 0.5, sub_frame['base_model'], linestyle = "-", color = "k", marker = 's', label = "Nugteren模型")
        
        ax.tick_params(axis='both', which='both', labelsize=10)
        ax.set_xlim(0, len(bench_index))
        ax.set_xticks(index + 0.5)
        ax.set_xticklabels(bench_index, rotation = 45)
        ##ax.set_xlabel('Kernel_nmaes', fontproperties = cn_font)
        ax.set_ylabel("时间开销（ms）", fontproperties = cn_font)
        ax.set_title(suite + "测试集", fontproperties = cn_font)

        if suite_id == num_suites:
            ax.legend(loc = 'upper left', fontsize = 'small', ncol = 2, bbox_to_anchor = (0, -0.45), prop = cn_font)

    fig = plt.gcf()
    fig.set_size_inches(6, 4)
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

def draw_footprint_v2(footprint_frame, title, save_path):
    suites = list(set(footprint_frame['suite'].values))
    suites.sort()

    num_suites = len(suites)
    suite_id = 0
    for suite in suites:
        suite_id = suite_id + 1
        ax = plt.subplot(num_suites, 1, suite_id)
        sub_frame = footprint_frame.ix[footprint_frame['suite'] == suite]

        bench_index = get_bench_index(sub_frame)
        index = np.arange(len(bench_index))

        ax.plot(index + 0.5, sub_frame['footprint'] / (1024 * 1024), linestyle = "-", color = "k", marker = "o", label = "Operation Intensity")
        
        ax.tick_params(axis='both', which='both', labelsize=10)
        ax.set_xlim(0, len(bench_index))
        ax.set_xticks(index + 0.5)
        ax.set_xticklabels(bench_index, rotation = 45)
        ##ax.set_xlabel('Kernel_nmaes', fontproperties = cn_font)
        ax.set_ylabel("内存印记（MB）", fontproperties = cn_font)
        ax.set_title(suite + "测试集", fontproperties = cn_font)

    fig = plt.gcf()
    fig.set_size_inches(6, 4)
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

def draw_op_intensity_v2(footprint_frame, title, save_path):
    suites = list(set(footprint_frame['suite'].values))
    suites.sort()

    num_suites = len(suites)
    suite_id = 0
    for suite in suites:
        suite_id = suite_id + 1
        ax = plt.subplot(num_suites, 1, suite_id)
        sub_frame = footprint_frame.ix[footprint_frame['suite'] == suite]

        bench_index = get_bench_index(sub_frame)
        index = np.arange(len(bench_index))

        ax.plot(index + 0.5, sub_frame['op_intensity'], linestyle = "-", color = "k", marker = "o", label = "Operation Intensity")
        
        ax.tick_params(axis='both', which='both', labelsize=10)
        ax.set_xlim(0, len(bench_index))
        ax.set_xticks(index + 0.5)
        ax.set_xticklabels(bench_index, rotation = 45)
        ##ax.set_xlabel('Kernel_nmaes', fontproperties = cn_font)
        ax.set_ylabel("运算访存比", fontproperties = cn_font)
        ax.set_title(suite + "测试集", fontproperties = cn_font)

    fig = plt.gcf()
    fig.set_size_inches(6, 4)
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

def draw_miss_rate_v2(footprint_frame, title, save_path):
    suites = list(set(footprint_frame['suite'].values))
    suites.sort()

    num_suites = len(suites)
    suite_id = 0
    for suite in suites:
        suite_id = suite_id + 1
        ax = plt.subplot(num_suites, 1, suite_id)
        sub_frame = footprint_frame.ix[footprint_frame['suite'] == suite]

        bench_index = get_bench_index(sub_frame)
        index = np.arange(len(bench_index))

        ax.plot(index + 0.5, sub_frame['profiler_miss'] * 100, linestyle = "-", color = "k", marker = "o", label = "Miss Rate")
        
        ax.tick_params(axis='both', which='both', labelsize=10)
        ax.set_xlim(0, len(bench_index))
        ax.set_xticks(index + 0.5)
        ax.set_xticklabels(bench_index, rotation = 45)
        ##ax.set_xlabel('Kernel_nmaes', fontproperties = cn_font)
        ax.set_ylabel("缓存缺失率（%）", fontproperties = cn_font)
        ax.set_title(suite + "测试集", fontproperties = cn_font)

    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(save_path)
    fig.clf()

def draw_miss_type(hit_count, comp_miss_count, uncomp_miss_count, save_path):
    index = np.arange(3)
    labels = ("命中", "强制型缺失", "非强制型缺失")
    d = (hit_count, comp_miss_count, uncomp_miss_count)

    rects = plt.bar(index + 0.1, d, 0.8, color = 'w')

    plt.xlim(0, 3)
    plt.ylim(0, max(d) * 1.2)
    plt.xticks(index + 0.5, labels, fontproperties = cn_font)
    #plt.xlabel('Reuse Distance', fontproperties = cn_font)
    plt.ylabel('统计次数', fontproperties = cn_font)

    fig = plt.gcf()
    fig.set_size_inches(6, 3)
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
    plt.xlabel('重用距离值', fontproperties = cn_font)
    plt.ylabel('出现次数', fontproperties = cn_font)

    fig = plt.gcf()
    fig.set_size_inches(6, 3)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(save_path)
    fig.clf()

def draw_pc_histo(pcs, pc_hit_counts, pc_miss_counts, save_path):
    index = np.arange(len(pcs))

    plt.bar(index + 0.1, pc_miss_counts, 0.4, color = 'w', label = "缺失数", hatch = "//")
    plt.bar(index + 0.5, pc_hit_counts, 0.4, color = 'w', label = "命中数", hatch = "..")

    plt.xlim(0, len(pcs))
    plt.ylim(0, max(max(pc_miss_counts), max(pc_hit_counts)) * 1.2)
    #plt.xticks(index + 0.5, distance_values, rotation = 'vertical')
    plt.xticks(index + 0.5, pcs)
    plt.title("-")
    plt.xlabel('程序计数器值（PC）', fontproperties = cn_font)
    plt.ylabel('Cache 缺失数/命中数', fontproperties = cn_font)
    plt.legend(loc = 'lower left', fontsize = 'small', ncol = 2, bbox_to_anchor = (0, 1), prop = cn_font)

    fig = plt.gcf()
    fig.set_size_inches(6, 3)
    fig.set_dpi(72)
    fig.set_tight_layout(True)
    fig.savefig(save_path)
    fig.clf()

def draw_miss_breakdown(miss_frame, save_path):
    kernels = miss_frame['kernel']
    index = np.arange(len(kernels))

    plt.bar(index + 0.15, miss_frame['model_comp_miss'], 0.7, label = 'compulsory miss', color = 'w', hatch = '//')
    plt.bar(index + 0.15, miss_frame['model_uncomp_miss'], 0.7, label = 'uncompulsory miss', color = 'w', hatch = '..', bottom = miss_frame['model_comp_miss'])

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

def draw_even_error(even_error_frame, title, save_path):
    index = np.arange(8)
    x_labels = ['jam_on', 'jam_off', 'stack_on', 'stack_off', 'trace_on', 'trace_off', 'latency_on', 'latency_off']


    plt.bar(0.2, even_error_frame['jam_on'] * 100, 0.8, color = 'w')
    plt.bar(1, even_error_frame['jam_off'] * 100, 0.8, color = 'w')
    plt.bar(2.2, even_error_frame['stack_on'] * 100, 0.8, color = 'w')
    plt.bar(3, even_error_frame['stack_off'] * 100, 0.8, color = 'w')
    plt.bar(4.2, even_error_frame['trace_on'] * 100, 0.8, color = 'w')
    plt.bar(5, even_error_frame['trace_off'] * 100, 0.8, color = 'w')
    plt.bar(6.2, even_error_frame['latency_on'] * 100, 0.8, color = 'w')
    plt.bar(7, even_error_frame['latency_off'] * 100, 0.8, color = 'w')

    ##  Do the plot
    plt.plot(index + 0.5, [even_error_frame['model'] * 100] * 8, label = 'model')
    plt.plot(index + 0.5, [even_error_frame['base_model'] * 100] * 8, label = 'base_model')

    ##  Figure options set
    plt.xlim(0, 8)
    plt.xticks(index + 0.5, x_labels)
    plt.title(title)
    #plt.xlabel('Kernel Name')
    plt.ylabel('even error(%)')
    plt.legend()

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
