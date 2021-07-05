import os
import csv
import time
import json
import toml
import argparse
import pandas as pd
import pygame
import numpy as np
from itertools import combinations

from overcooked_ai_pcg import LSI_CONFIG_EXP_DIR, LSI_LOG_DIR, LSI_CONFIG_ALGO_DIR, LSI_CONFIG_MAP_DIR, LSI_CONFIG_AGENT_DIR
from overcooked_ai_pcg.helper import run_overcooked_game, read_in_lsi_config
from overcooked_ai_pcg.LSI.qd_algorithms import Individual, FeatureMap
from overcooked_ai_pcg.helper import read_in_lsi_config, init_env_and_agent
 
def search_extreme_cells(elite_map, features):
    feature_combs = combinations(range(0,len(features)), 2)
    f_extreme_bound = []
    for i in range(len(feature_combs)):
        f_extreme_bound.append([[np.inf, np.inf], [np.inf, -np.inf], [-np.inf, np.inf], [-np.inf, -np.inf]]) # four extreme corners [(0,0), (0, max), (max, 0), (max, max)]
    f_extreme_bound = np.array(f_extreme_bound) # lenght should be the amount of 2d plots
    print('f_extreme_bound shape =', f_extreme_bound.shape())

    for elite in elite_map:
        splited = elite.split(":")
        curr_row_idx = int(splited[0])
        curr_col_idx = int(splited[1])
        curr_mat_idx = int(splited[2])

        for i in feature_comb in enumerate(feature_combs): # for each 2d plot, find 4 corners
            curr_row = int(splited[feature_comb[0]])
            curr_col = int(splited[feature_comb[1]])

            # find (0,0)
            low_left_row, low_left_col = f_extreme_bound[i][0]
            if curr_row < low_left_row:
                low_left_row = curr_row
                low_left_col = curr_col
            elif curr_row == low_left_row:
                if curr_col < low_left_col:
                    low_left_row = curr_row
                    low_left_col = curr_col
            f_extreme_bound[i][0][0] = low_left_row
            f_extreme_bound[i][0][1] = low_left_col  

            # find (0,max)
            low_left_row, up_left_col = f_extreme_bound[i][1]
            if curr_row < low_left_row:
                low_left_row = curr_row
                up_left_col = curr_col
            elif curr_row == low_left_row:
                if curr_col > up_left_col:
                    low_left_row = curr_row
                    up_left_col = curr_col
            f_extreme_bound[i][1][0] = low_left_row
            f_extreme_bound[i][1][1] = up_left_col  

            # find (max, 0)
            up_row, low_col = f_extreme_bound[i][2]
            if curr_col < low_col:
                up_row = curr_row
                low_col = curr_col
            elif curr_col == low_col:
                if curr_row > up_row:
                    up_row = curr_row
                    low_col = curr_col
            f_extreme_bound[i][2][0] = up_row
            f_extreme_bound[i][2][1] = low_col  




def retrieve_k_individuals(experiment_config, elite_map_config, agent_configs, individuals, row_idx, col_idx, mat_idx, log_dir, k):

    #from IPython import embed
    #embed()

    num_simulations = experiment_config["num_simulations"
    ]
    feature_map = FeatureMap(
        num_simulations,
        feature_ranges=[(bc["low"], bc["high"])
                        for bc in elite_map_config["Map"]["Features"]],
        resolutions=[
            bc["resolution"] for bc in elite_map_config["Map"]["Features"]
        ],
    )
    feature0_name = features[0]["name"]
    feature1_name = features[1]["name"]
    feature2_name = features[2]["name"]


    num_individuals = num_simulations * 51
    relevant_individuals = []
    for indx in range(num_individuals):
        individual = individuals.iloc[indx]
        if indx % 1000== 0:
          print(indx)

        if (np.isnan(individual["ID"])==False): 
             feature0 = individual[feature0_name]
             feature1 = individual[feature1_name]
             feature2 = individual[feature2_name]
             ind = Individual()
             ind.features = ([feature0], [feature1], [feature2])
             #from IPython import embed
             #em3bed()

             ind.fitness = individual["fitness"]
             index = feature_map.get_index(ind)

             if index[0] == row_idx and index[1] == col_idx and index[2] == mat_idx: 
                relevant_individuals.append(individual)

    sorted_individuals = sorted(relevant_individuals, key=lambda x: x.fitness)[::-1]
    sorted_individuals = sorted_individuals[:k]

    IDs = []
    for dd in range(k):
      IDs.append(sorted_individuals[dd]["ID"])

    
    return IDs, sorted_individuals


def play_individual(individual, agent_configs):
    ind = Individual()
    ind.level = individual.lvl_str
    ind.rand_seed = int(individual.rand_seed)
    agent1, agent2, env, mdp = init_env_and_agent(ind, agent_configs[-1])

    for agent_config in agent_configs:
        fitness, _, _, _, ind.joint_actions, _, _ = run_overcooked_game(ind, agent_config, render=True)
    return

    print("Fitness: {}; Total sparse reward: {};".format(ind.fitness, ind.scores))
    print("Checkpoints", ind.checkpoints)
    print("Workloads:", ind.player_workloads, "; Concurrently active:", ind.concurr_active, "; Stuck time:", ind.stuck_time)

    return


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

        if curr_row_idx == row_idx and curr_col_idx == col_idx and curr_mat_idx == mat_idx:


            ind_id = int(splited[3])*51
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
    experiment_config, _, elite_map_config, agent_configs = read_in_lsi_config(opt.config)

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


    IDs, individuals  = retrieve_k_individuals(experiment_config, elite_map_config, agent_configs, individuals, row_idx, col_idx, mat_idx, log_dir,3)

    from IPython import embed
    embed()
    
    for individual in individuals:
      play_individual(individual, agent_configs)



    #play(elite_map, agent_configs, individuals, row_idx, col_idx, mat_idx, log_dir)
