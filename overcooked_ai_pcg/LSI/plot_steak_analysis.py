import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, argparse
from overcooked_ai_pcg import (LSI_STEAK_STUDY_RESULT_DIR,
                               LSI_STEAK_STUDY_CONFIG_DIR,
                               LSI_STEAK_STUDY_AGENT_DIR)

def plot_kb_2(data, log_dir, log_name):
    img_dir = os.path.join(log_dir, log_name)
    for i, name in enumerate(['freq', 'avg']):
        # Sample data (replace these with your actual data)
        mid_1_0 = np.array(data['Mid-1_120_not_aware_3'])[:,i]
        mid_1_1 = np.array(data['Mid-1-120_3'])[:,i]

        mid_2_0 = np.array(data['Mid-2_120_not_aware_3'])[:,i]
        mid_2_1 = np.array(data['Mid-2-120_3'])[:,i]

        none_3_0 = np.array(data['None-3_120_not_aware_3'])[:,i]
        none_3_1 = np.array(data['None-3-120_3'])[:,i]

        
        # Define the number of bins and their range
        num_bins = 12  # You can adjust this to change the number of bins

        # Create subplots with grouped histograms in a single row
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the grouped histogram for Data 1, Data 2, and Data 3 in each subplot
        for j, [data1, data2, map_name] in enumerate([[mid_1_0, mid_1_1, 'Mid-1'], [mid_2_0, mid_2_1, 'Mid-2'], [none_3_0, none_3_1, 'None-3']]):
            bin_range = (min(min(data1), min(data2)), max(max(data1), max(data2)))

            # Calculate the histograms manually
            hist_data = [np.histogram(d, bins=num_bins, range=bin_range)[0] for d in [data1, data2]]

            # Create a histogram with three distinct bars in each bin
            bin_centers = np.arange(bin_range[0], bin_range[1], (bin_range[1] - bin_range[0]) / num_bins)
            bar_width = (bin_centers[1] - bin_centers[0]) / 4  # Adjust the width as needed

            for k, (d, color, label) in enumerate(zip(hist_data, ['red', 'green'], ['Not Aware', 'Aware'])):
                axes[j].bar(bin_centers - bar_width * k, d, width=bar_width, color=color, label=label)
                # axes[i].hist(d, bins=num_bins, range=bin_range, alpha=0.5, color=color, label=label, edgecolor='k', stacked=False)
                axes[j].set_xlabel(' '.join(['Knowledge base differences', name+'.']))
                axes[j].set_ylabel('Number of participants')
                axes[j].set_title(f'Knowledge base differences {name}. in {map_name}')
                axes[j].legend()

        # Adjust layout
        plt.tight_layout()
        plt.savefig(img_dir+'_kb_'+name+'.png')

        # Show the plot
        # plt.show()

def plot_kb(data, log_dir, log_name):
    img_dir = os.path.join(log_dir, log_name)
    for i, name in enumerate(['freq', 'avg']):
        # Sample data (replace these with your actual data)
        mid_1_0 = np.array(data['Mid-1_120_not_aware_3'])[:,i]
        mid_1_1 = np.array(data['Mid-1-120_3'])[:,i]
        mid_1_2 = np.array(data['Mid-1-120_0'])[:,i]

        mid_2_0 = np.array(data['Mid-2_120_not_aware_3'])[:,i]
        mid_2_1 = np.array(data['Mid-2-120_3'])[:,i]
        mid_2_2 = np.array(data['Mid-2-120_0'])[:,i]

        none_3_0 = np.array(data['None-3_120_not_aware_3'])[:,i]
        none_3_1 = np.array(data['None-3-120_3'])[:,i]
        none_3_2 = np.array(data['None-3-120_0'])[:,i]

        
        # Define the number of bins and their range
        num_bins = 12  # You can adjust this to change the number of bins

        # Create subplots with grouped histograms in a single row
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the grouped histogram for Data 1, Data 2, and Data 3 in each subplot
        for j, [data1, data2, data3, map_name] in enumerate([[mid_1_0, mid_1_1, mid_1_2, 'Mid-1'], [mid_2_0, mid_2_1, mid_2_2, 'Mid-2'], [none_3_0, none_3_1, none_3_2, 'None-3']]):
            bin_range = (min(min(data1), min(data2), min(data3)), max(max(data1), max(data2), max(data3)))

            # Calculate the histograms manually
            hist_data = [np.histogram(d, bins=num_bins, range=bin_range)[0] for d in [data1, data2, data3]]

            # Create a histogram with three distinct bars in each bin
            bin_centers = np.arange(bin_range[0], bin_range[1], (bin_range[1] - bin_range[0]) / num_bins)
            bar_width = (bin_centers[1] - bin_centers[0]) / 4  # Adjust the width as needed

            for k, (d, color, label) in enumerate(zip(hist_data, ['red', 'green', 'blue'], ['Not Aware', 'Aware', 'Aware Not Act'])):
                axes[j].bar(bin_centers - bar_width * k, d, width=bar_width, color=color, label=label)
                # axes[i].hist(d, bins=num_bins, range=bin_range, alpha=0.5, color=color, label=label, edgecolor='k', stacked=False)
                axes[j].set_xlabel('_'.join(['Knowledge base differences', name]))
                axes[j].set_ylabel('Number of participants')
                axes[j].set_title(f'Knowledge base differences {name} in {map_name}')
                axes[j].legend()

        # Adjust layout
        plt.tight_layout()
        plt.savefig(img_dir+'_kb_'+name+'.png')

        # Show the plot
        # plt.show()

def plot_workload_2(data, log_dir, log_name):
    img_dir = os.path.join(log_dir, log_name)
    for i, name in enumerate(['prep', 'plating']):
        # Sample data (replace these with your actual data)
        mid_1_0 = np.array(data['Mid-1_120_not_aware_3'])[:,i]
        mid_1_1 = np.array(data['Mid-1-120_3'])[:,i]

        mid_2_0 = np.array(data['Mid-2_120_not_aware_3'])[:,i]
        mid_2_1 = np.array(data['Mid-2-120_3'])[:,i]

        none_3_0 = np.array(data['None-3_120_not_aware_3'])[:,i]
        none_3_1 = np.array(data['None-3-120_3'])[:,i]

        # Define the number of bins and their range
        num_bins = 12  # You can adjust this to change the number of bins

        # Create subplots with grouped histograms in a single row
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the grouped histogram for Data 1, Data 2, and Data 3 in each subplot
        for j, [data1, data2, map_name] in enumerate([[mid_1_0, mid_1_1, 'Mid-1'], [mid_2_0, mid_2_1, 'Mid-2'], [none_3_0, none_3_1, 'None-3']]):
            # bin_range = (min(min(data1), min(data2), min(data3)), max(max(data1), max(data2), max(data3)))
            bin_range = (-6, 6)

            # Calculate the histograms manually
            hist_data = [np.histogram(d, bins=num_bins, range=bin_range)[0] for d in [data1, data2]]

            # Create a histogram with three distinct bars in each bin
            bin_centers = np.arange(bin_range[0], bin_range[1], (bin_range[1] - bin_range[0]) / num_bins)
            bar_width = (bin_centers[1] - bin_centers[0]) / 4  # Adjust the width as needed

            for k, (d, color, label) in enumerate(zip(hist_data, ['red', 'green'], ['Not Aware', 'Aware'])):
                axes[j].bar(bin_centers - bar_width * k, d, width=bar_width, color=color, label=label)
                axes[j].set_xlabel('_'.join(['Workload', name]))
                axes[j].set_ylabel('Number of participants')
                axes[j].set_title(f'Workload {name} in {map_name}')
                axes[j].legend()

        # Adjust layout
        plt.tight_layout()
        plt.savefig(img_dir+'_workload_'+name+'.png')

        # Show the plot
        # plt.show()

def plot_workload(data, log_dir, log_name):
    img_dir = os.path.join(log_dir, log_name)
    for i, name in enumerate(['prep', 'plating']):
        # Sample data (replace these with your actual data)
        mid_1_0 = np.array(data['Mid-1_120_not_aware_3'])[:,i]
        mid_1_1 = np.array(data['Mid-1-120_3'])[:,i]
        mid_1_2 = np.array(data['Mid-1-120_0'])[:,i]

        mid_2_0 = np.array(data['Mid-2_120_not_aware_3'])[:,i]
        mid_2_1 = np.array(data['Mid-2-120_3'])[:,i]
        mid_2_2 = np.array(data['Mid-2-120_0'])[:,i]

        none_3_0 = np.array(data['None-3_120_not_aware_3'])[:,i]
        none_3_1 = np.array(data['None-3-120_3'])[:,i]
        none_3_2 = np.array(data['None-3-120_0'])[:,i]

        # Define the number of bins and their range
        num_bins = 12  # You can adjust this to change the number of bins

        # Create subplots with grouped histograms in a single row
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the grouped histogram for Data 1, Data 2, and Data 3 in each subplot
        for j, [data1, data2, data3, map_name] in enumerate([[mid_1_0, mid_1_1, mid_1_2, 'Mid-1'], [mid_2_0, mid_2_1, mid_2_2, 'Mid-2'], [none_3_0, none_3_1, none_3_2, 'None-3']]):
            # bin_range = (min(min(data1), min(data2), min(data3)), max(max(data1), max(data2), max(data3)))
            bin_range = (-6, 6)

            # Calculate the histograms manually
            hist_data = [np.histogram(d, bins=num_bins, range=bin_range)[0] for d in [data1, data2, data3]]

            # Create a histogram with three distinct bars in each bin
            bin_centers = np.arange(bin_range[0], bin_range[1], (bin_range[1] - bin_range[0]) / num_bins)
            bar_width = (bin_centers[1] - bin_centers[0]) / 4  # Adjust the width as needed

            for k, (d, color, label) in enumerate(zip(hist_data, ['red', 'green', 'blue'], ['Not Aware', 'Aware', 'Aware Not Act'])):
                axes[j].bar(bin_centers - bar_width * k, d, width=bar_width, color=color, label=label)
                axes[j].set_xlabel('_'.join(['Workload', name]))
                axes[j].set_ylabel('Number of participants')
                axes[j].set_title(f'Workload {name} in {map_name}')
                axes[j].legend()

        # Adjust layout
        plt.tight_layout()
        plt.savefig(img_dir+'_workload_'+name+'.png')

        # Show the plot
        # plt.show()

def plot_fluency(data, log_dir, log_name):
    img_dir = os.path.join(log_dir, log_name)

    # Sample data (replace these with your actual data)
    mid_1_0 = data['Mid-1_120_not_aware_3']
    mid_1_1 = data['Mid-1-120_3']
    mid_1_2 = data['Mid-1-120_0']

    mid_2_0 = data['Mid-2_120_not_aware_3']
    mid_2_1 = data['Mid-2-120_3']
    mid_2_2 = data['Mid-2-120_0']

    none_3_0 = data['None-3_120_not_aware_3']
    none_3_1 = data['None-3-120_3']
    none_3_2 = data['None-3-120_0']

    # Define the number of bins and their range
    num_bins = 12  # You can adjust this to change the number of bins

    # Create subplots with grouped histograms in a single row
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the grouped histogram for Data 1, Data 2, and Data 3 in each subplot
    for i, [data1, data2, data3, map_name] in enumerate([[mid_1_0, mid_1_1, mid_1_2, 'Mid-1'], [mid_2_0, mid_2_1, mid_2_2, 'Mid-2'], [none_3_0, none_3_1, none_3_2, 'None-3']]):
        # bin_range = (min(min(data1), min(data2), min(data3)), max(max(data1), max(data2), max(data3)))
        bin_range = (0,20)
        # Calculate the histograms manually
        hist_data = [np.histogram(d, bins=num_bins, range=bin_range)[0] for d in [data1, data2, data3]]

        # Create a histogram with three distinct bars in each bin
        bin_centers = np.arange(bin_range[0], bin_range[1], (bin_range[1] - bin_range[0]) / num_bins)
        bar_width = (bin_centers[1] - bin_centers[0]) / 4  # Adjust the width as needed

        for k, (d, color, label) in enumerate(zip(hist_data, ['red', 'green', 'blue'], ['Not Aware', 'Aware', 'Aware Not Act'])):
            axes[i].bar(bin_centers - bar_width * k, d, width=bar_width, color=color, label=label)
            axes[i].set_xlabel('Interrupted Frequency (%)')
            axes[i].set_ylabel('Number of participants')
            # axes[i].set_xlim([0,20])
            axes[i].set_title(f'Fluency measurement in {map_name}')
            axes[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig(img_dir+'_fluency.png')

    # Show the plot
    # plt.show()

def plot_fluency_2(data, log_dir, log_name):
    img_dir = os.path.join(log_dir, log_name)

    # Sample data (replace these with your actual data)
    mid_1_0 = data['Mid-1_120_not_aware_3']
    mid_1_1 = data['Mid-1-120_3']

    mid_2_0 = data['Mid-2_120_not_aware_3']
    mid_2_1 = data['Mid-2-120_3']

    none_3_0 = data['None-3_120_not_aware_3']
    none_3_1 = data['None-3-120_3']

    # Define the number of bins and their range
    num_bins = 12  # You can adjust this to change the number of bins

    # Create subplots with grouped histograms in a single row
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the grouped histogram for Data 1, Data 2, and Data 3 in each subplot
    for i, [data1, data2, map_name] in enumerate([[mid_1_0, mid_1_1, 'Mid-1'], [mid_2_0, mid_2_1, 'Mid-2'], [none_3_0, none_3_1, 'None-3']]):
        # bin_range = (min(min(data1), min(data2)), max(max(data1), max(data2)))
        bin_range = (0,20)
        # Calculate the histograms manually
        hist_data = [np.histogram(d, bins=num_bins, range=bin_range)[0] for d in [data1, data2]]

        # Create a histogram with three distinct bars in each bin
        bin_centers = np.arange(bin_range[0], bin_range[1], (bin_range[1] - bin_range[0]) / num_bins)
        bar_width = (bin_centers[1] - bin_centers[0]) / 4  # Adjust the width as needed

        for k, (d, color, label) in enumerate(zip(hist_data, ['red', 'green'], ['Not Aware', 'Aware'])):
            axes[i].bar(bin_centers - bar_width * k, d, width=bar_width, color=color, label=label)
            axes[i].set_xlabel('Interrupted Frequency (%)')
            axes[i].set_ylabel('Number of participants')
            axes[i].set_title(f'Fluency measurement in {map_name}')
            axes[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig(img_dir+'_fluency.png')

    # Show the plot
    # plt.show()

def load_analyzed_data():
    human_log_csv = os.path.join(LSI_STEAK_STUDY_RESULT_DIR,
                                 "analysis_log.csv")
    if not os.path.exists(human_log_csv):
        print("Log dir does not exit.")
        exit(1)
    human_log_data = pd.read_csv(human_log_csv)
    return human_log_csv, human_log_data

def read_in_data():
    human_log_csv, human_log_data = load_analyzed_data()

    # workload: prep count vs plating
    workload = {} # -6 to 6

    # fluency: 
    fluency = {} # 0 to 20 frequency

    # knowledge gap
    kb_gap = {}

    for index, info in human_log_data.iterrows():
        if info["lvl_type"] not in workload.keys():
            workload[info["lvl_type"]] = [[info["prep_count_diff"], info["plating_count_diff"]]]
        else:
            workload[info["lvl_type"]] += [[info["prep_count_diff"], info["plating_count_diff"]]]

        if info["lvl_type"] not in fluency.keys():
            fluency[info["lvl_type"]] = [info["interrupt_freq"]]
        else:
            fluency[info["lvl_type"]] += [info["interrupt_freq"]]

        if info["lvl_type"] not in kb_gap.keys():
            kb_gap[info["lvl_type"]] = [[info["kb_diff_freq"], info["kb_diff_avg"]]]
        else:
            kb_gap[info["lvl_type"]] += [[info["kb_diff_freq"], info["kb_diff_avg"]]]

    return workload, fluency, kb_gap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-l',
    #                     '--log_index',
    #                     help='Integer: index of the study log',
    #                     required=False,
    #                     default=-1)
    opt = parser.parse_args()

    workload, fluency, kb_gap = read_in_data()
    # plot_fluency(fluency, LSI_STEAK_STUDY_RESULT_DIR, log_name='fluency')
    # plot_workload(workload, LSI_STEAK_STUDY_RESULT_DIR, log_name='workload')
    # plot_kb(kb_gap, LSI_STEAK_STUDY_RESULT_DIR, log_name='kb')

    plot_fluency_2(fluency, LSI_STEAK_STUDY_RESULT_DIR, log_name='fluency_2')
    plot_workload_2(workload, LSI_STEAK_STUDY_RESULT_DIR, log_name='workload_2')
    plot_kb_2(kb_gap, LSI_STEAK_STUDY_RESULT_DIR, log_name='kb_2')