import os
import csv
import time
import json
import toml
import argparse
import pandas as pd
from overcooked_ai_pcg import LSI_CONFIG_EXP_DIR, LSI_LOG_DIR, LSI_CONFIG_ALGO_DIR, LSI_CONFIG_MAP_DIR, LSI_CONFIG_AGENT_DIR
from overcooked_ai_pcg.helper import run_overcooked_game, read_in_lsi_config
from overcooked_ai_pcg.LSI.qd_algorithms import Individual

# def read_in_lsi_config(exp_config_file):
#     experiment_config = toml.load(exp_config_file)
#     algorithm_config = toml.load(
#         os.path.join(LSI_CONFIG_ALGO_DIR,
#                      experiment_config["experiment_config"]["algorithm_config"]))
#     elite_map_config = toml.load(
#         os.path.join(LSI_CONFIG_MAP_DIR,
#                      experiment_config["experiment_config"]["elite_map_config"]))
#     agent_configs = []
#     for agent_config_file in experiment_config["agent_config"]:
#         agent_config = toml.load(
#         os.path.join(LSI_CONFIG_AGENT_DIR, experiment_config["experiment_config"]["agent_config"]))
#         agent_configs.append(agent_config)
#     return experiment_config, algorithm_config, elite_map_config, agent_configs


def log_actions(ind, agent_config, log_dir, f1, f2, row_idx, col_idx, ind_id):
    agent1_config = agent_config["Agent1"]
    agent2_config = agent_config["Agent2"]

    log_file = ""

    if agent1_config["name"] == "fixed_plan_agent" and agent2_config[
            "name"] == "fixed_plan_agent":
        log_file = "fixed_plan"

    elif agent1_config["name"] == "preferenced_human" and agent2_config[
            "name"] == "human_aware_agent":
        log_file = "human_aware"

    elif agent1_config["name"] == "mdp_agent" and agent2_config[
            "name"] == "greedy_agent":
        log_file = "mdp_"

    elif agent1_config["name"] == "qmdp_agent" and agent2_config[
            "name"] == "greedy_agent":
        log_file = "qmdp_"

    log_file += ("_joint_actions_"+str(f1)+"_"+str(f2)+"_"+str(row_idx)+"_"+str(col_idx)+"_"+str(ind_id)+".json")
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
         log_dir, ind_id):
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
    ind_id = int(ind_id)
    ind_id_index = ind_id*51
    lvl_str = individuals["lvl_str"][ind_id_index]
    print("Playing in individual %d" % ind_id)
    print(lvl_str)
    ind = Individual()
    ind.level = lvl_str
    ind.human_preference = individuals["human_preference"][ind_id_index]
    ind.human_adaptiveness = individuals["human_adaptiveness"][ind_id_index]
    ind.rand_seed = int(individuals["rand_seed"][ind_id_index])
    print(ind.rand_seed)
    for agent_config in agent_configs:
        print(agent_config["Agent1"], agent_config["Agent2"])
        fitness, _, _, _, ind.joint_actions, _, _ = run_overcooked_game(ind, agent_config, render=True)
        print("Fitness:", fitness)
        log_actions(ind, agent_config, log_dir, f1, f2, row_idx, col_idx, ind_id)
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
    parser.add_argument('-id',
                        '--ind_id',
                        help='id of the individual',
                        required=False,
                        default=1)
    parser.add_argument('-m',
                        '--mode',
                        help='analyze heat map mode',
                        required=False,
                        default=False)

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
         log_dir, opt.ind_id)
