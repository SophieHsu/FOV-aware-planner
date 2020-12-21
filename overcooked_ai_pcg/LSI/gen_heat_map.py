"""Generates heat maps of the archives for the paper.

Pass in the path LOGDIR to a logging directory created by the search. This
script will read in data and configuration information from the logging
directory, and for each pair of features in the map, it will generate the
following files in LOGDIR/images:

- PDF called `map_final_{y_idx}_{x_idx}.pdf` showing the final heatmap. This is
  a PDF because PDF figures work better with Latex.
- Video called `map_video_{y_idx}_{x_idx}.avi` showing the progress of the
  heatmap.
- GIF called `map_gif_{y_idx}_{x_idx}.gif` showing the progress of the heatmap.

The {y_idx} and {x_idx} are the indices of the features used in the file in the
list `elite_map_config.Map.Features` in `config.toml`. {y_idx} is the index of
the feature along the y-axis, and {x_idx} is the index of the feature used along
the x-axis.

Usage:
    python gen_heat_map.py -l LOGDIR
"""
import argparse
import csv
import os
import shutil
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from overcooked_ai_pcg import LSI_CONFIG_ALGO_DIR, LSI_CONFIG_MAP_DIR

# Visualization settings.
REGULAR_FIGSIZE = (7.5, 6)
FPS = 30  # FPS for video and gif.
NUM_TICKS = 5  # Number of ticks on plots.
ANIMATION_INTERVAL = 200  # ms between each frame of the video and gif.
COLORMAP = "viridis"  # Colormap for everything.

# Map settings.
FITNESS_MIN = 0
FITNESS_MAX = 20000
WORKLOAD_DIFFS_LOW = np.array([-6, -2, -2])
WORKLOAD_DIFFS_HIGH = np.array([6, 2, 2])


def read_in_lsi_config(exp_config_file):
    experiment_config = toml.load(exp_config_file)
    algorithm_config = toml.load(
        os.path.join(
            LSI_CONFIG_ALGO_DIR,
            experiment_config["experiment_config"]["algorithm_config"]))
    elite_map_config = toml.load(
        os.path.join(
            LSI_CONFIG_MAP_DIR,
            experiment_config["experiment_config"]["elite_map_config"]))
    return experiment_config, algorithm_config, elite_map_config


def csv_data_to_pandas(data, y_feature_idx, x_feature_idx, num_features,
                       is_workloads_diff):
    """Converts one row from elite_map.csv into a dataframe.

    (One row of the data contains a snapshot of the entire map at that
    iteration).

    If is_workloads_diff is True, this function will instead return a dict of
    dataframes mapping values along one of the BCs to dataframes with data of
    elites with that BC.
    """
    map_dims = tuple(map(int, data[0].split('x')))
    elites = data[1:]  # The rest of the row contains all the data.

    # Create 2D dataframe(s) to store the map data.
    if is_workloads_diff:
        # For workload diff, create a dict of pandas dataframes; each one has y
        # as index and x as columns, and the dict is indexed by our 3rd BC.
        dataframes = {}

        # Index along which we will enumerate the BC. It is the one index not
        # covered by y_feature_idx and x_feature_idx.
        enumerate_idx = list(set(range(3)) - {y_feature_idx, x_feature_idx})[0]

        # Index is descending to make the heatmap look better.
        index_labels = np.arange(WORKLOAD_DIFFS_HIGH[y_feature_idx],
                                 WORKLOAD_DIFFS_LOW[y_feature_idx] - 1, -1)
        column_labels = np.arange(WORKLOAD_DIFFS_LOW[x_feature_idx],
                                  WORKLOAD_DIFFS_HIGH[x_feature_idx] + 1)
        initial_data = np.full((len(index_labels), len(column_labels)), np.nan)
        for i in range(WORKLOAD_DIFFS_LOW[enumerate_idx],
                       WORKLOAD_DIFFS_HIGH[enumerate_idx] + 1):
            # Make sure to copy the initial data -- it seems pandas keeps a
            # reference to it.
            dataframes[i] = pd.DataFrame(np.copy(initial_data), index_labels,
                                         column_labels)
    else:
        # Create a pandas dataframe with our two BCs on the indices and columns.
        # Index is descending to make the heatmap look better.
        index_labels = np.arange(map_dims[y_feature_idx] - 1, -1, -1)
        column_labels = np.arange(0, map_dims[x_feature_idx])
        initial_data = np.full((len(index_labels), len(column_labels)), np.nan)
        dataframe = pd.DataFrame(initial_data, index_labels, column_labels)

    # Iterate through the entries in the map and insert them into the
    # appropriate dict.
    for elite in elites:
        # Do some pre-processing (see recordList) -- parse the data, normalize
        # the fitness.
        tokens = elite.split(":")  # Each elite starts in string format.
        bc_indices = np.array(list(map(int, tokens[:num_features])))
        cell_y = bc_indices[y_feature_idx]
        cell_x = bc_indices[x_feature_idx]

        # Adjust BCs only for workloads diff.
        if is_workloads_diff:
            cell_y += WORKLOAD_DIFFS_LOW[y_feature_idx]
            cell_x += WORKLOAD_DIFFS_LOW[x_feature_idx]
            cell_enum = bc_indices[enumerate_idx] + WORKLOAD_DIFFS_LOW[
                enumerate_idx]

        # Adjust fitness.
        fitness = float(tokens[num_features + 1])
        if fitness == 1:
            fitness = 0
        elif 200_000 <= fitness < 400_000:
            fitness -= 200_000
        elif fitness >= 400_000:
            fitness -= 390_000

        if is_workloads_diff:
            # Insert into the correct dict. We keep all vals (none should be
            # intersecting).
            dataframes[cell_enum].loc[cell_y, cell_x] = fitness
        else:
            # Insert into the dataframe. Override with better vals.
            old_fitness = dataframe.loc[cell_y, cell_x]
            if np.isnan(old_fitness) or fitness > old_fitness:
                dataframe.loc[cell_y, cell_x] = fitness

    return dataframes if is_workloads_diff else dataframe


def create_axes(is_workloads_diff, dataframe):
    """Creates a figure, axis/axes, and colorbar axis.

    If is_workloads_diff is True, the ax returned will be an array of axes rather
    than a single axis.
    """
    if is_workloads_diff:
        y_len = len(dataframe[list(dataframe)[0]].index)
        x_len = len(dataframe[list(dataframe)[0]].columns)
        is_vertical = y_len > x_len

        # Make the figure wider for horizontal and square plots.
        figsize = (9, 6) if is_vertical else (15, 3)
        if y_len == x_len:
            figsize = (18, 3)

        # Colorbar needs a bit more proportion in horizontal plots to get same
        # height.
        height_ratios = [0.95, 0.05] if is_vertical else [0.9, 0.1]

        fig = plt.figure(figsize=figsize)
        num_plots = len(dataframe)  # dataframe is a dict in this case.
        spec = fig.add_gridspec(ncols=num_plots,
                                nrows=2,
                                height_ratios=height_ratios)
        ax = np.array([fig.add_subplot(spec[0, i]) for i in range(num_plots)],
                      dtype=object)

        # Make the colorbar span the entire figure in vertical plots and only
        # the middle three plots in horizontal figures.
        cbar_ax = fig.add_subplot(spec[1, :] if is_vertical else spec[
            1, num_plots // 2 - 1:num_plots // 2 + 2])

    else:
        fig, ax = plt.subplots(1, 1, figsize=REGULAR_FIGSIZE)
        ax_divider = make_axes_locatable(ax)
        cbar_ax = ax_divider.append_axes("right", size="7%", pad="2%")

    return fig, ax, cbar_ax


def plot_heatmap(dataframe, ax, cbar_ax, img_title, y_name, x_name,
                 is_workloads_diff):
    """Plots a heatmap of the given dataframe onto the given ax.

    A colorbar is created on cbar_ax.

    If is_workloads_diff is True, ax should be an array of axes on which to plot
    the heatmap for each value of the enumerating BC. dataframe should then be a
    dict as described in csv_data_to_pandas().

    enum_bc_name is only provided if is_workload_diff is True.
    """
    if is_workloads_diff:
        for idx, entry in enumerate(dataframe.items()):
            enum_bc, ind_dataframe = entry
            sns.heatmap(
                ind_dataframe,
                annot=False,
                cmap=COLORMAP,
                fmt=".0f",
                xticklabels=len(ind_dataframe.columns) // NUM_TICKS + 1,
                yticklabels=(len(ind_dataframe.index) // NUM_TICKS +
                             1 if idx == 0 else False),
                vmin=FITNESS_MIN,
                vmax=FITNESS_MAX,
                square=True,
                ax=ax[idx],
                cbar=idx == 0,  # Only plot cbar for first plot.
                cbar_ax=cbar_ax,
                cbar_kws={"orientation": "horizontal"})
            ax[idx].set_title(f"{enum_bc}", pad=12)
            if idx == 0:  # y-label on first plot.
                ax[idx].set_ylabel(y_name, labelpad=12)
            if idx == len(dataframe) // 2:  # x-label on center plot.
                ax[idx].set_xlabel(x_name, labelpad=10)
        ax[0].figure.tight_layout()
    else:
        sns.heatmap(dataframe,
                    annot=False,
                    cmap=COLORMAP,
                    fmt=".0f",
                    xticklabels=len(dataframe.columns) // NUM_TICKS + 1,
                    yticklabels=len(dataframe.index) // NUM_TICKS + 1,
                    vmin=FITNESS_MIN,
                    vmax=FITNESS_MAX,
                    square=True,
                    ax=ax,
                    cbar_ax=cbar_ax)
        ax.set_title(img_title, pad=12)
        ax.set_ylabel(y_name, labelpad=12)
        ax.set_xlabel(x_name, labelpad=10)
        ax.figure.tight_layout()


def main(opt):
    # Read in configurations.
    experiment_config, algorithm_config, elite_map_config = read_in_lsi_config(
        os.path.join(opt.logdir, "config.tml"))
    features = elite_map_config['Map']['Features']
    is_workloads_diff = (experiment_config["experiment_config"]
                         ["elite_map_config"] == "workloads_diff.tml")
    img_title = algorithm_config["name"]

    # Global plot settings.
    sns.set_theme(
        context="paper",
        style="darkgrid",
        font="sans-serif",
        font_scale=1.5,
        rc={
            # Refer to https://matplotlib.org/3.2.1/tutorials/introductory/customizing.html
            "axes.facecolor": ".9",
            "xtick.bottom": True,
            "xtick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.left": True,
            "ytick.major.width": 0.8,
            "ytick.major.size": 3.0,
        })

    # Create image directory or clear out previous images.
    img_dir = os.path.join(opt.logdir, "images/")
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.mkdir(img_dir)

    # Retrieve elite map. This file contains data about the _entire_ map after
    # each evaluation.
    with open(os.path.join(opt.logdir, "elite_map.csv"), "r") as csvfile:
        elite_map_data = list(csv.reader(csvfile, delimiter=','))
        elite_map_data = elite_map_data[1:]  # Exclude header.

    # Create outputs for every pair of features (allow reversing so we can get
    # all orientations of images).
    for y_feature_idx, y_feature in enumerate(features):
        for x_feature_idx, x_feature in enumerate(features):
            if y_feature_idx == x_feature_idx:
                continue

            y_name, x_name = y_feature["name"], x_feature["name"]
            if is_workloads_diff:
                # The index of the feature along which to enumerate BCs.
                enumerate_idx = list(
                    set(range(3)) - {y_feature_idx, x_feature_idx})[0]
                enumerate_name = features[enumerate_idx]["name"]
            else:
                enumerate_name = None

            print("-------------------------\n"
                  "## Info ##\n"
                  f"y: Feature {y_feature_idx} ({y_name})\n"
                  f"x: Feature {x_feature_idx} ({x_name})\n"
                  "## Saving PDF of final map ##")
            dataframe = csv_data_to_pandas(elite_map_data[-1],
                                           y_feature_idx, x_feature_idx,
                                           len(features), is_workloads_diff)
            fig, ax, cbar_ax = create_axes(is_workloads_diff, dataframe)
            plot_heatmap(dataframe, ax, cbar_ax, img_title, y_name, x_name,
                         is_workloads_diff)
            ax[0].figure.suptitle(f"{img_title} | {enumerate_name}")
            fig.savefig(
                os.path.join(img_dir,
                             f"map_final_{y_feature_idx}_{x_feature_idx}.pdf"))

            if not opt.video:
                continue

            print("## Generating video and gif ##")
            fig, ax, cbar_ax = create_axes(is_workloads_diff, dataframe)
            ax[0].figure.suptitle(f"{img_title} | {enumerate_name}")

            def draw_frame(frame):
                """Draws the heatmap for each frame of the animation."""
                # pylint: disable = cell-var-from-loop
                print(".", end="")
                sys.stdout.flush()

                # Clear axes to make room for new plots.
                if isinstance(ax, np.ndarray):
                    for a in ax.ravel():
                        a.clear()
                else:
                    ax.clear()
                cbar_ax.clear()

                dataframe = csv_data_to_pandas(elite_map_data[frame - 1],
                                               y_feature_idx, x_feature_idx,
                                               len(features), is_workloads_diff)
                plot_heatmap(dataframe, ax, cbar_ax, img_title, y_name, x_name,
                             is_workloads_diff)

            anim = animation.FuncAnimation(
                fig,
                draw_frame,
                frames=np.append(
                    np.arange(opt.step_size,
                              len(elite_map_data) + 1, opt.step_size),
                    np.full(5, len(elite_map_data))),
                repeat=True,
                interval=ANIMATION_INTERVAL,
            )

            anim.save(
                os.path.join(img_dir,
                             f"map_video_{y_feature_idx}_{x_feature_idx}.avi"))
            anim.save(
                os.path.join(img_dir,
                             f"map_gif_{y_feature_idx}_{x_feature_idx}.gif"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-l",
        "--logdir",
        help=("Path to experiment logging directory. Images are"
              "also output here in the 'images' subdirectory"),
        required=True,
    )
    parser.add_argument(
        "-s",
        "--step_size",
        help="step size of the animation to generate",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--video",
        dest="video",
        action="store_true",
        default=True,
        help=("Whether to create the video (it may be useful to turn this off "
              "for debugging. Pass --no-video to disable."),
    )
    parser.add_argument("--no-video", dest="video", action="store_false")
    parser.set_defaults(video=True)

    main(parser.parse_args())
