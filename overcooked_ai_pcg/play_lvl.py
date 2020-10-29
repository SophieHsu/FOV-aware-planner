import os
import csv
import time
import json
import argparse
import pandas as pd
from overcooked_ai_pcg import LSI_CONFIG_EXP_DIR, LSI_LOG_DIR
from overcooked_ai_pcg.helper import run_overcooked_game, read_in_lsi_config
from overcooked_ai_pcg.LSI.qd_algorithms import Individual


def log_actions(ind, agent_config, log_dir):
    agent1_config = agent_config["Agent1"]
    agent2_config = agent_config["Agent2"]

    log_file = ""

    if agent1_config["name"] == "fixed_plan_agent" and agent2_config[
            "name"] == "fixed_plan_agent":
        log_file = "fixed_plan"

    elif agent1_config["name"] == "preferenced_human" and agent2_config[
            "name"] == "human_aware_agent":
        log_file = "human_aware"

    log_file += "_joint_actions.json"
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


def play(elite_map, agent_configs, individuals, f1, f2, row_idx, col_idx,
         log_dir):
    """
    Find the individual in the specified cell in the elite map
    and run overcooked game with the specified agents

    Args:
        elite_map (list): list of logged cell strings.
                          See elite_map.csv for detail.
        agent_configs: toml config object of agents
        individuals (pd.dataFrame): all individuals logged
        f1, f2 (int, int): index of the features to use
        row_idx, col_idx (int, int): index of the cell in the elite map
    """
    for elite in elite_map:
        splited = elite.split(":")
        curr_row_idx = int(splited[f1])
        curr_col_idx = int(splited[f2])
        if curr_row_idx == row_idx and curr_col_idx == col_idx:
            ind_id = int(splited[num_features])
            lvl_str = individuals["lvl_str"][ind_id]
            print("Playing in individual %d" % ind_id)
            print(lvl_str)
            ind = Individual()
            ind.level = lvl_str
            ind.human_preference = individuals["human_preference"][ind_id]
            ind.human_adaptiveness = individuals["human_adaptiveness"][ind_id]
            ind.rand_seed = individuals["rand_seed"][ind_id]
            for agent_config in agent_configs:
                fitness, _, _, _, _ = run_overcooked_game(ind, agent_config, render=False)
                print("Fitness: %d" % fitness)
                log_actions(ind, agent_config, log_dir)
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

    opt = parser.parse_args()

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
    num_row = features[f1]['resolution']
    num_col = features[f2]['resolution']
    row_idx = int(opt.row_idx)
    col_idx = int(opt.col_idx)
    assert (row_idx < num_row)
    assert (col_idx < num_col)

    play(elite_map, agent_configs, individuals, f1, f2, row_idx, col_idx,
         log_dir)
