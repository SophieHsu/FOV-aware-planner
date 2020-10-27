import numpy as np
import os
import csv
import toml
import argparse
import time
import ast 
from itertools import product, combinations
from overcooked_ai_pcg import LSI_IMAGE_DIR, LSI_LOG_DIR
from overcooked_ai_pcg.helper import read_in_lsi_config, run_overcooked_game
from overcooked_ai_pcg.LSI.evaluator import calculate_bc
from overcooked_ai_pcg.LSI.qd_algorithms import Individual

def read_in_ind_log(log_path):
    with open(log_path, 'r') as f:
        # Read all the data from the csv file
        print(log_path)
        allRows = list(csv.reader(f, delimiter=','))

    return allRows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        help='path to the experiment config file',
                        required=True)
    parser.add_argument('-i',
                        '--idx',
                        help='index of the individual',
                        required=True,
                        default=0)
    parser.add_argument('-l',
                        '--log_file',
                        help='filepath to the elite map log file',
                        required=True,
                        default=os.path.join(LSI_LOG_DIR, "individuals_log.csv"))
    opt = parser.parse_args()

    # read in the name of the algorithm and features to plot
    experiment_config, algorithm_config, elite_map_config, agent_config = read_in_lsi_config(
        opt.config)
    features = elite_map_config['Map']['Features']
    inds = read_in_ind_log(opt.log_file)
    
    ind = Individual()
    idx = int(opt.idx)
    ind.ID = int(inds[idx][0])
    ind.fitness = int(inds[idx][1])
    ind.score = int(inds[idx][2])
    ind.player_workload = ast.literal_eval(inds[idx][-5])
    ind.human_preference = float(inds[idx][-4])
    ind.human_adaptiveness = float(inds[idx][-3])
    ind.rand_seed = int(inds[idx][-2])
    ind.level = inds[idx][-1]

    print(ind.ID, ind.fitness, '\n', ind.player_workload, '\n', ind.level)

    new_ind = Individual()
    start_time = time.time()
    new_ind.fitness, new_ind.score, new_ind.checkpoints, new_ind.player_workload = run_overcooked_game(ind, ind.level, agent_config, render=True, worker_id=0)
    _, new_ind = calculate_bc(0, new_ind, elite_map_config)

    print(new_ind.ID, new_ind.fitness, new_ind.score, new_ind.player_workload, new_ind.checkpoints)