import os
import sys
import csv
import toml
import json
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from overcooked_ai_pcg.helper import run_overcooked_game
from overcooked_ai_pcg.gen_lvl import generate_lvl
from overcooked_ai_pcg.LSI import bc_calculate
from overcooked_ai_pcg.LSI.qd_algorithms import *
from overcooked_ai_pcg import LSI_LOG_DIR, LSI_CONFIG_DIR, LSI_CONFIG_TRIAL_DIR, LSI_CONFIG_MAP_DIR, LSI_CONFIG_ALGO_DIR


def eval_overcooked(ind, visualize, elite_map_config):
    """
    Evaluate overcooked game by running a game and calculate relevant bc

    Args:
        ind (Individual): individual instance
        visualize (bool): render the game or not
        elite_map_config: toml config object of the feature maps
    """
    # run game and get fitness
    print("Start evaluation...")
    fitness = run_overcooked_game(ind.level, render=visualize)

    # calculate bc out of the game
    ind.features = []
    for bc in elite_map_config["Map"]["Features"]:
        # get the function the calculate bc
        bc_fn_name = bc["name"]
        bc_fn = getattr(bc_calculate, bc_fn_name)
        bc_val = bc_fn(ind)
        ind.features.append(bc_val)
    ind.features = tuple(ind.features)
    print("Game end; fitness =", fitness)
    return fitness

evaluate = eval_overcooked

def run_trial(num_to_evaluate,
              algorithm_name,
              algorithm_config,
              elite_map_config,
              trial_name,
              model_path,
              visualize):
    """
    Run a single trial from the experiment.

    Args:
        num_to_evaluate (int): total number of evaluations of QD algorithm to run.
        algorithm_name (string): name of the QD algorithm to use
        algorithm_config: toml config object of QD algorithm
        elite_map_config: toml config object of the feature maps
        trial_name (string): name of the trial
        model_path (string): file path to the GAN model
        visualize (bool): render the game or not
    """

    # get ranges of the bc
    feature_ranges = []
    for bc in elite_map_config["Map"]["Features"]:
        feature_ranges.append((bc["low"],bc["high"]))

    # instantiate feature map
    if(trial_name.split('_')[1] == "demo"):
        feature_map = FeatureMap(num_to_evaluate, feature_ranges, resolutions=(5, 5))
    else:
        sys.exit('unknown BC name. Exiting the program.')

    # instantiate algorithm
    if algorithm_name == "MAPELITES":
        print("Start Running MAPELITES")
        mutation_power = algorithm_config["mutation_power"]
        initial_population = algorithm_config["initial_population"]
        algorithm_instance = MapElitesAlgorithm(mutation_power,
                                                initial_population,
                                                num_to_evaluate,
                                                feature_map,)
    elif algorithm_name=="RANDOM":
        print("Start Running RANDOM")
        algorithm_instance=RandomGenerator(num_to_evaluate,
                                           feature_map,)

    # run search
    simulation = 1
    while algorithm_instance.is_running():
        ind = algorithm_instance.generate_individual()

        ind.level = generate_lvl(1, model_path, ind.param_vector)
        ind.fitness = evaluate(ind, visualize, elite_map_config)

        algorithm_instance.return_evaluated_individual(ind)

        print(str(simulation)+"/"+str(num_to_evaluate)+" simulations finished")
        simulation += 1

    # algorithm_instance.all_records.to_csv(
    #     os.path.join(LSI_LOG_DIR, trial_name+"_all_simulations.csv"))

def start_search(sim_number,
                 trial_index,
                 experiment_toml,
                 model_path,
                 visualize):
    """
    Read in relevant config files of the trial and run it.

    Args:
        sim_number (int): index of the worker
        trial_index (int): index of the trial
        experiment_toml: toml config object of the experiment
        model_path (string): file path to the GAN model
        visualize (bool): render the game run or not
    """
    trial = experiment_toml["Trials"][trial_index]
    trial_toml = toml.load(os.path.join(LSI_CONFIG_TRIAL_DIR, trial["trial_config"]))
    num_simulations = trial_toml["num_simulations"]
    algorithm_to_run = trial_toml["algorithm"]
    algorithm_config = toml.load(os.path.join(LSI_CONFIG_ALGO_DIR, trial_toml["algorithm_config"]))
    elite_map_config = toml.load(os.path.join(LSI_CONFIG_MAP_DIR, trial_toml["elite_map_config"]))
    trial_name = trial_toml["trial_name"] + "_sim" + str(sim_number)
    run_trial(num_simulations,
              algorithm_to_run,
              algorithm_config,
              elite_map_config,
              trial_name,
              model_path,
              visualize)
    print("Finished One Trial")