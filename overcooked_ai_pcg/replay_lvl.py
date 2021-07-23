"""Replays an Overcooked level.

Images showing frames of the level are saved in the logdir in a directory called
`replay_...` (the rest of the name includes the parameters of the replay). To
compose these frames into a video, use `ffmpeg` within that directory, such as:

    ffmpeg -r 6 -i "%*.png" video.mp4

The 6 above is the frame rate (6 fps).
"""
import argparse
import csv
import json
import os
import shutil
import time

import pandas as pd
import toml

from overcooked_ai_pcg import (LSI_CONFIG_AGENT_DIR, LSI_CONFIG_ALGO_DIR,
                               LSI_CONFIG_EXP_DIR, LSI_CONFIG_MAP_DIR,
                               LSI_LOG_DIR)
from overcooked_ai_pcg.helper import (read_in_lsi_config, run_overcooked_game,
                                      visualize_lvl)
from overcooked_ai_pcg.LSI.qd_algorithms import Individual


def log_actions(ind, agent_config, log_dir, f1, f2, row_idx, col_idx, ind_id):
    agent1_config = agent_config["Agent1"]
    agent2_config = agent_config["Agent2"]

    log_file = ""

    if agent1_config["name"] == "fixed_plan_agent" and agent2_config[
            "name"] == "fixed_plan_agent":
        log_file = "fixed_plan_"

    elif agent1_config["name"] == "preferenced_human" and agent2_config[
            "name"] == "human_aware_agent":
        log_file = "human_aware_"

    elif agent1_config["name"] == "mdp_agent" and agent2_config[
            "name"] == "greedy_agent":
        log_file = "mdp_"

    elif agent1_config["name"] == "qmdp_agent" and agent2_config[
            "name"] == "greedy_agent":
        log_file = "qmdp_"

    log_file += ("joint_actions_" + str(f1) + "_" + str(f2) + "_" +
                 str(row_idx) + "_" + str(col_idx) + "_" + str(ind_id) +
                 ".json")

    full_path = os.path.join(log_dir, log_file)
    if os.path.exists(full_path):
        print("Joint actions logged before, skipping...")
        return

    # log the joint actions if not logged before
    with open(full_path, "w") as f:
        json.dump({
            "joint_actions": ind.joint_actions,
            "lvl_str": ind.level,
        }, f)
        print("Joint actions saved")


def play_ind_id(elite_map,
                agent_configs,
                individuals,
                f1,
                f2,
                row_idx,
                col_idx,
                log_dir,
                ind_id,
                num_sim=1):
    ind_id = int(ind_id)
    ind_id_index = ind_id * num_sim
    lvl_str = individuals["lvl_str"][ind_id_index]
    print("Playing in individual %d" % ind_id)
    print(lvl_str)
    ind = Individual()
    ind.level = lvl_str
    # ind.human_preference = individuals["human_preference"][ind_id_index]
    # ind.human_adaptiveness = individuals["human_adaptiveness"][ind_id_index]
    ind.rand_seed = 0#int(individuals["rand_seed"][ind_id_index])
    for agent_config in agent_configs:
        print(agent_config["Agent1"], agent_config["Agent2"])
        fitness, _, _, _, ind.joint_actions, _, _ = run_overcooked_game(
            ind, agent_config, render=True)
        print("Fitness:", fitness)
        log_actions(ind, agent_config, log_dir, f1, f2, row_idx, col_idx,
                    ind_id)

    visualize_lvl(
        lvl_str, log_dir, "rendered_level_" + str(f1) + "_" + str(f2) + "_" +
        str(row_idx) + "_" + str(col_idx) + "_" + str(ind_id) + ".png")
    return


def play(elite_map,
         agent_configs,
         individuals,
         log_dir,
         row_idx,
         col_idx,
         mat_idx=None,
         is_3d=False,
         mode="replay",
         num_sim=1):
    """Find the individual in the specified cell in the elite map and run
    overcooked game with the specified agents.

    Args:
        elite_map (list): list of logged cell strings.
                          See elite_map.csv for detail.
        agent_configs: toml config object of agents
        individuals (pd.dataFrame): all individuals logged
        f1, f2 (int, int): index of the features to use
        row_idx, col_idx, mat_idx (int, int, int): index of the cell in the
            elite map
        is_3d: Boolean indicating whether the map is 3D.
        mode (str): "render" or "replay"
            for "render": the script merely render the level
            for "replay": the script rerun the game
    """
    for elite in elite_map:
        splited = elite.split(":")
        curr_row_idx = int(splited[0])
        curr_col_idx = int(splited[1])
        curr_mat_idx = int(splited[2]) if is_3d else None
        curr_idx = (curr_row_idx, curr_col_idx, curr_mat_idx)
        # print(curr_idx)
        if curr_idx == (row_idx, col_idx, mat_idx):
            ind_id = int(splited[num_features])
            ind_idx = ind_id * num_sim
            lvl_str = individuals[individuals["ID"] ==
                                  ind_id]["lvl_str"].iloc[0]
            print("Playing in individual %d" % ind_id)
            print(lvl_str)
            ind = Individual()
            ind.level = lvl_str
            ind.human_preference = individuals["human_preference"][ind_idx]
            ind.human_adaptiveness = individuals["human_adaptiveness"][ind_idx]
            ind.rand_seed = int(individuals["rand_seed"][ind_idx])

            if mode == "replay":
                for agent_idx, agent_config in enumerate(agent_configs):
                    # Create image directory -- removes existing one.
                    img_dir = os.path.join(
                        log_dir, (f"replay_f1{f1}_f2{f2}_row{row_idx}_"
                                  f"col{col_idx}_ind{ind_id}_agent{agent_idx}"))
                    if os.path.exists(img_dir):
                        shutil.rmtree(img_dir)
                    os.mkdir(img_dir)
                    print(f"Saving video frames in {img_dir}")

                    fitness, _, _, _, ind.joint_actions, _, _ = run_overcooked_game(
                        ind,
                        agent_config,
                        render=True,
                        track_belief=True,
                        img_name=(
                            lambda timestep: f"{img_dir}/{timestep:05d}.png"),
                    )
                    print("Fitness: ", fitness)
                    log_actions(ind, agent_config, log_dir, f1, f2, row_idx,
                                col_idx, ind_id)

                visualize_lvl(
                    lvl_str, log_dir,
                    "rendered_level_" + str(row_idx) + "_" + str(col_idx) +
                    "_" + str(mat_idx) + "_" + str(ind_id) + ".png")
            elif mode == "render":
                visualize_lvl(
                    lvl_str, log_dir,
                    f"rendered_level_{row_idx}_{col_idx}_{ind_id}.png")
                # log_actions(ind, agent_config, log_dir, row_idx,
                #             col_idx, ind_id)
            return

    print("No individual found in the specified cell")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        help='path of experiment config file',
                        required=True)
    parser.add_argument('-l',
                        '--log_dir',
                        help='path of log directory',
                        required=True)
    parser.add_argument('-row',
                        '--row_idx',
                        help='index f1 in elite map',
                        required=True)
    parser.add_argument('-col',
                        '--col_idx',
                        help='index f2 in elite map',
                        required=True)
    parser.add_argument('-mat',
                        '--matrix_idx',
                        help='index f3 in elite map. If this is passed in, the\
                        script would attempt to get individual from 3D archive.\
                        ',
                        required=False)
    parser.add_argument('-mode',
                        help="mode to replay or merely render the level.",
                        required=False,
                        default="replay")
    parser.add_argument('-id',
                        '--ind_id',
                        help='id of the individual',
                        required=False,
                        default=1)
    parser.add_argument('-f1',
                        '--feature1_idx',
                        help='index of the first feature to be used',
                        required=False,
                        default=0)
    parser.add_argument('-f2',
                        '--feature2_idx',
                        help='index of the second feature to be used',
                        required=False,
                        default=1)
    parser.add_argument('-sim',
                        '--num_sim',
                        help='Number of simulations done for one level',
                        required=False,
                        default=1)

    opt = parser.parse_args()

    is_3d = True
    if opt.matrix_idx is None:
        is_3d = False
    # read in full elite map
    log_dir = opt.log_dir
    elite_map_log_file = os.path.join(log_dir, "elite_map.csv")
    elite_map_log = open(elite_map_log_file, 'r')
    all_rows = list(csv.reader(elite_map_log, delimiter=','))
    elite_map = all_rows[-1][1:]
    elite_map_log.close()

    # read in individuals
    individuals = pd.read_csv(os.path.join(log_dir, "individuals_log.csv"))

    # read in configs
    _, _, elite_map_config, agent_configs = read_in_lsi_config(opt.config)

    # read in feature index
    features = elite_map_config['Map']['Features']
    num_features = len(features)
    f1 = int(opt.feature1_idx)
    f2 = int(opt.feature2_idx)
    assert (f1 < num_features)
    assert (f2 < num_features)

    # read in row/col index
    # num_row = features[0]['resolution']
    # num_col = features[1]['resolution']
    # num_mat = features[2]['resolution']
    row_idx = int(opt.row_idx)
    col_idx = int(opt.col_idx)
    # assert (row_idx < num_row)
    # assert (col_idx < num_col)

    # number of simulations for one level map to row number in log files
    num_sim = 1
    if int(opt.num_sim) > 1:
        num_sim = int(opt.num_sim) + 1  # extra row for logging average/mode

    # play_ind_id(elite_map, agent_configs, individuals, f1, f2, row_idx, col_idx, log_dir, opt.ind_id, num_sim)
    mat_idx = None
    if is_3d:
        mat_idx = int(opt.matrix_idx)
        # assert (mat_idx < num_mat)

    play(elite_map,
         agent_configs,
         individuals,
         log_dir,
         row_idx,
         col_idx,
         mat_idx=mat_idx,
         is_3d=is_3d,
         mode=opt.mode,
         num_sim=num_sim)
