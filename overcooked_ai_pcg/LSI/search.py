import os
import sys
import csv
import toml
import json
import pandas as pd
import numpy as np
import torch
import time
from torch.autograd import Variable
from collections import OrderedDict
from multiprocessing.managers import BaseManager
from overcooked_ai_pcg.helper import run_overcooked_game
from overcooked_ai_pcg.gen_lvl import generate_lvl, generate_rnd_lvl
from overcooked_ai_pcg.LSI import bc_calculate
from overcooked_ai_pcg.LSI.qd_algorithms import *
from overcooked_ai_pcg.LSI.evaluator import *
from overcooked_ai_pcg import LSI_LOG_DIR, LSI_CONFIG_DIR, LSI_CONFIG_TRIAL_DIR, LSI_CONFIG_MAP_DIR, LSI_CONFIG_ALGO_DIR


# def eval_overcooked(ind, visualize, elite_map_config):
#     """
#     Evaluate overcooked game by running a game and calculate relevant bc

#     Args:
#         ind (Individual): individual instance
#         visualize (bool): render the game or not
#         elite_map_config: toml config object of the feature maps
#     """
#     # run game and get fitness
#     print("Start evaluation...")
#     fitness = run_overcooked_game(ind.level, render=visualize)

#     # calculate bc out of the game
#     ind.features = []
#     for bc in elite_map_config["Map"]["Features"]:
#         # get the function the calculate bc
#         bc_fn_name = bc["name"]
#         bc_fn = getattr(bc_calculate, bc_fn_name)
#         bc_val = bc_fn(ind)
#         ind.features.append(bc_val)
#     ind.features = tuple(ind.features)
#     print("Game end; fitness =", fitness)
#     return fitness

# multiprocessing related
global evaluators_list, idle_workers, running_workers, worker_list
evaluators_list = []
idle_workers = []
running_workers = []
worker_list = []

def assign_eval_task(ind, worker_id, sim_id):
    worker_list[worker_id].set_ind(ind)
    worker_list[worker_id].set_sim_id(sim_id)
    worker_list[worker_id].set_status(Status.EVALUATING)

def init_workers(visualize, elite_map_config, num_cores, model_path):
    BaseManager.register('Worker', Worker)
    manager = BaseManager()
    manager.start()
    # starting process
    for id in range(0, num_cores):
        worker = manager.Worker(id)
        worker_list.append(worker)
        idle_workers.append(worker.get_id())
        evaluator = OvercookedEvaluator(id, worker,
                                        visualize,
                                        elite_map_config,
                                        model_path)
        evaluator.start()
        evaluators_list.append(evaluator)

def worker_has_finished(worker_id):
    if worker_list[worker_id].get_status() == Status.IDLE:
        return True
    else:
        return False

def terminate_all_workers(num_cores):
    for id in range(num_cores):
        worker_list[id].set_status(Status.TERMINATING)
        evaluators_list[id].join()

def run_trial(num_to_evaluate,
              algorithm_name,
              algorithm_config,
              elite_map_config,
              trial_name,
              model_path,
              visualize,
              num_cores):
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
        num_cores (int): number of processes to run
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
    start_time = time.time()
    simulation = 1
    init_workers(visualize, elite_map_config, num_cores, model_path)
    while algorithm_instance.is_running():
        # print("Looking for idle workers")

        # assign job to idle workers
        while len(idle_workers) > 0 and simulation <= num_to_evaluate:
            print("idle workers: " + str(idle_workers))
            ind = algorithm_instance.generate_individual()
            print("Starting simulation: %d/%d on worker %d"
                % (simulation, num_to_evaluate, idle_workers[0]))

            worker_id = idle_workers.pop(0)
            assign_eval_task(ind, worker_id, simulation)
            running_workers.append(worker_id)

            simulation += 1

        # find done workers
        num_running_workers = len(running_workers)
        for _ in range(num_running_workers):
            worker_id = running_workers.pop(0)
            if worker_has_finished(worker_id):
                worker = worker_list[worker_id]
                evaluated_ind = worker.get_ind()
                algorithm_instance.return_evaluated_individual(evaluated_ind)
                idle_workers.append(worker_id)
                print("Finished simulation: %d/%d"
                      % (worker.get_sim_id(), num_to_evaluate))
            else:
                running_workers.append(worker_id)
        time.sleep(1)

    finish_time = time.time()
    print("Total time: " + str(finish_time - start_time) + " seconds")


def start_search(sim_number,
                 trial_index,
                 experiment_toml,
                 model_path,
                 visualize,
                 num_cores):
    """
    Read in relevant config files of the trial and run it.

    Args:
        sim_number (int): index of the worker
        trial_index (int): index of the trial
        experiment_toml: toml config object of the experiment
        model_path (string): file path to the GAN model
        visualize (bool): render the game run or not
        num_cores (int): number of processes to run
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
              visualize,
              num_cores)
    print("Finished One Trial")
    terminate_all_workers(num_cores)