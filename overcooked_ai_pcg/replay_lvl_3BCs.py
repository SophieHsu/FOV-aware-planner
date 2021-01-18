import os
import csv
import time
import json
import toml
import argparse
import pandas as pd
import pygame
import numpy as np

from overcooked_ai_pcg import LSI_CONFIG_EXP_DIR, LSI_LOG_DIR, LSI_CONFIG_ALGO_DIR, LSI_CONFIG_MAP_DIR, LSI_CONFIG_AGENT_DIR
from overcooked_ai_pcg.helper import run_overcooked_game, read_in_lsi_config
from overcooked_ai_pcg.LSI.qd_algorithms import Individual
from overcooked_ai_pcg.helper import read_in_lsi_config, init_env_and_agent


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




def play(elite_map, agent_configs, individuals, row_idx, col_idx, mat_idx, log_dir):
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
        curr_row_idx = int(splited[0])
        curr_col_idx = int(splited[1])
        curr_mat_idx = int(splited[2])

        ind_id = int(splited[3])
        if ind_id == 1886:
            pass


        if curr_row_idx == row_idx and curr_col_idx == col_idx and curr_mat_idx == mat_idx:


            ind_id = int(splited[3])*52
            lvl_str = individuals["lvl_str"][ind_id]
            print("Playing in individual %d" % ind_id)
            print(lvl_str)


            ind = Individual()
            ind.level = lvl_str
            ind.human_preference = individuals["human_preference"][ind_id]
            ind.human_adaptiveness = individuals["human_adaptiveness"][ind_id]
            ind.rand_seed = int(individuals["rand_seed"][ind_id])

            agent1, agent2, env, mdp = init_env_and_agent(ind, agent_configs[-1])

            #theApp = App(env, agent1, agent2, rand_seed=ind.rand_seed, player_idx=0, slow_time=False)
            #ind.fitness, ind.scores, ind.checkpoints, ind.player_workloads, ind.joint_actions, ind.concurr_active, ind.stuck_time = theApp.on_execute()
            for agent_config in agent_configs:
                fitness, _, _, _, ind.joint_actions, _, _ = run_overcooked_game(ind, agent_config, render=True)
                #from IPython import embed
                #embed()
                #print("Fitness: %d" % fitness[0])
                #log_actions(ind, agent_config, log_dir, f1, f2, row_idx, col_idx, ind_id)
            return

            print("Fitness: {}; Total sparse reward: {};".format(ind.fitness, ind.scores))
            print("Checkpoints", ind.checkpoints)
            print("Workloads:", ind.player_workloads, "; Concurrently active:", ind.concurr_active, "; Stuck time:", ind.stuck_time)

            #log_actions(ind, agent_configs[-1], log_dir, f1, f2, row_idx, col_idx, ind_id)

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

    parser.add_argument('-matrix',
                        '--mat_idx',
                        help='index f3 in elite map',
                        required=True)

    parser.add_argument('-id',
                        '--ind_id',
                        help='id of the individual',
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
    #num_features = len(features)
    #f1 = int(opt.feature1_idx)
    #f2 = int(opt.feature2_idx)
    #assert (f1 < num_features)
    #assert (f2 < num_features)

    # read in row/col index
    num_row = features[0]['resolution']
    num_col = features[1]['resolution']
    num_mat = features[2]['resolution']
    row_idx = int(opt.row_idx)
    col_idx = int(opt.col_idx)
    mat_idx = int(opt.mat_idx)



    assert (row_idx < num_row)
    assert (col_idx < num_col)
    assert (mat_idx < num_mat)

    play(elite_map, agent_configs, individuals, row_idx, col_idx, mat_idx, log_dir)
