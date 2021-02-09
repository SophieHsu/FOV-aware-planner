import os
import ast
import pandas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
# from ss_plotting.make_plots import plot_bar_graph
from overcooked_ai_pcg import LSI_HUMAN_STUDY_RESULT_DIR

from overcooked_ai_pcg.LSI.human_study import (DETAILED_STUDY_TYPES,
                                               load_human_log_data)

plot_bc_names = [
    "workloads",
    "concurr_active",
    "stuck_time",
]

categories = [
    "uneven_workloads",
    "even_workloads",
    "low_team_fluency",
    "high_team_fluency"
]

workloads = np.zeros((2,3,13), dtype=int) # uneven, even
team_fluency = np.zeros((2,2,101), dtype=int) # low, high

def log_bc_plot(lvl_types, bc_name, human_data, normalize_data=[]):
    ingd_offset = 6

    for i, value in human_data.iteritems():
        category_idx = categories.index(lvl_types[i].split("-")[0])

        if bc_name == "workloads" and category_idx<2:
            value = ast.literal_eval(value)

            ingd_idx = (value[0]['num_ingre_held'] - value[1]['num_ingre_held'] + ingd_offset)
            workloads[category_idx%2, 0, ingd_idx] += 1
            
            plate_idx = (value[0]['num_plate_held'] - value[1]['num_plate_held'] + ingd_offset)
            workloads[category_idx%2, 1, plate_idx] += 1
            
            order_idx = (value[0]['num_served'] - value[1]['num_served'] + ingd_offset)
            workloads[category_idx%2, 2, order_idx] += 1
            
        elif bc_name == "concurr_active" and category_idx >= 2:
            team_fluency[category_idx%2, 0, min(int((value/normalize_data[i])*100), 100)] += 1
        
        elif bc_name == "stuck_time" and category_idx >= 2:
            team_fluency[category_idx%2, 1, min(int((value/normalize_data[i])*100), 100)] += 1
        
        else:
            pass


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        if rect.get_height() > 0:
            height = rect.get_height() 
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

def plot_bcs(num_idx, human_study_log_dir):
    bar_width = 0.3
    label_loc = 0
    for i, category in enumerate(categories):
        fig, ax = plt.subplots()

        if i < 2:
            label_loc = np.arange(workloads.shape[2])
            rects_ingd = ax.bar(label_loc-bar_width, workloads[i, 0], bar_width, label="diff_ingredient")
            rects_plate = ax.bar(label_loc, workloads[i, 1], bar_width, label="diff_plate")
            rects_order = ax.bar(label_loc+bar_width, workloads[i, 2], bar_width, label="diff_order")

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('Number of levels')
            ax.set_ylim([0,num_idx])
            ax.set_title(category.replace("_", " ").capitalize())
            ax.set_xlabel('Workload differences')
            ax.set_xticks(label_loc)
            ax.set_xticklabels(list(np.arange(-6,7,1)))
            ax.legend()

            autolabel(ax, rects_ingd)
            autolabel(ax, rects_plate)
            autolabel(ax, rects_order)

            # plot_bar_graph([workloads[i, 0], workloads[i, 1], workloads[i, 2]], ["blue", "green", "orange"],series_labels=["diff_ingredient", "diff_plate", "diff_order"], plot_ylabel='Number of people', plot_title=category.replace("_", " ").capitalize())

        else:
            label_loc = np.arange(team_fluency.shape[2])
            rects_concurr = ax.bar(label_loc-bar_width/2, team_fluency[i%2, 0], bar_width, label="concurr_active")
            rects_stuck = ax.bar(label_loc+bar_width/2, team_fluency[i%2, 1], bar_width, label="time_stuck")

            ax.set_ylabel('Number of levels')
            ax.set_ylim([0,num_idx])
            ax.set_title(category.replace("_", " ").capitalize())
            ax.set_xlabel('Team fluency BCs(%)')
            # ax.set_xticks(label_loc)
            # ax.set_xticklabels(labels)
            ax.legend()
            
            autolabel(ax, rects_concurr)
            autolabel(ax, rects_stuck)

        fig.tight_layout()
        # plt.show()
        plt.savefig(human_study_log_dir+'/'+category+'.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--log_dir',
                        help='path to human study result log directory',
                        required=False,
                        default=None)

    opt = parser.parse_args()
    human_study_log_dir = LSI_HUMAN_STUDY_RESULT_DIR
    if opt.log_dir is not None:
        human_study_log_dir = opt.log_dir
    for i, log_index in enumerate(sorted(os.listdir(human_study_log_dir))):
        if not os.path.isdir(
                os.path.join(LSI_HUMAN_STUDY_RESULT_DIR, log_index)):
            continue
        _, human_log_data = load_human_log_data(log_index)
        human_log_data = human_log_data.sort_values(by=["lvl_type"])

        log_bc_plot(human_log_data["lvl_type"], "workloads", human_log_data.loc[:,f"workloads"])
        log_bc_plot(human_log_data["lvl_type"], "concurr_active", human_log_data.loc[:,f"concurr_active"], [ast.literal_eval(v)[-1] for _,v in human_log_data.loc[:,f"checkpoints"].iteritems()])
        log_bc_plot(human_log_data["lvl_type"], "stuck_time", human_log_data.loc[:,f"stuck_time"], [ast.literal_eval(v)[-1] for _,v in human_log_data.loc[:,f"checkpoints"].iteritems()])

    plot_bcs(len(next(os.walk(human_study_log_dir))[1])*3, human_study_log_dir)