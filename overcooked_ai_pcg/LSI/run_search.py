import os
import toml
import time
import argparse
from collections import OrderedDict
from multiprocessing.managers import BaseManager
from overcooked_ai_pcg.LSI.logger import *
from overcooked_ai_pcg.LSI.qd_algorithms import *
from overcooked_ai_pcg.LSI.evaluator import *
from overcooked_ai_pcg import GAN_TRAINING_DIR, LSI_CONFIG_EXP_DIR, LSI_CONFIG_ALGO_DIR, LSI_CONFIG_MAP_DIR


# multiprocessing related variables
global evaluators_list, idle_workers, running_workers, worker_list
evaluators_list = []
idle_workers = []
running_workers = []
worker_list = []

def assign_eval_task(ind, worker_id, sim_id):
    """
    Assign evaluation task to a worker by setting relevant variables

    Args:
        ind (Individual): Individual instance to be evaluated
        worker_id (int): id of the worker
        sim_id (int): index of the simulation/evaluation job
    """
    worker_list[worker_id].set_ind(ind)
    worker_list[worker_id].set_sim_id(sim_id)
    worker_list[worker_id].set_status(Status.EVALUATING)

def init_workers(visualize, elite_map_config, num_cores, model_path):
    """
    Initialize specified number of workers and processes

    Args:
        visualize (bool): render the game or not
        elite_map_config: toml config object of the feature maps
        num_cores (int): number of processes to run
        model_path (string): file path to the GAN model
    """
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
    """
    Determine whether the worker has finished evaluating

    Args:
        worker_id (int): id of the worker
    """
    if worker_list[worker_id].get_status() == Status.IDLE:
        return True
    else:
        return False

def terminate_all_workers(num_cores):
    """
    Appropriately terminate all workers/processes

    Args:
        num_cores (int): number of processes to run
    """
    for id in range(num_cores):
        worker_list[id].set_status(Status.TERMINATING)
        evaluators_list[id].join()

def search(num_simulations,
           algorithm_config,
           elite_map_config,
           model_path,
           visualize,
           num_cores):
    """
    Run search with the specified algorithm and elite map

    Args:
        num_simulations (int): total number of evaluations of QD algorithm to run.
        algorithm_config: toml config object of QD algorithm
        elite_map_config: toml config object of the feature maps
        model_path (string): file path to the GAN model
        visualize (bool): render the game or not
        num_cores (int): number of processes to run
    """

    # config feature map
    feature_ranges = []
    resolutions = []
    for bc in elite_map_config["Map"]["Features"]:
        feature_ranges.append((bc["low"],bc["high"]))
        resolutions.append(bc["resolution"])
    feature_map = FeatureMap(num_simulations, feature_ranges, resolutions)

    # config algorithm instance
    algorithm_name = algorithm_config["name"]
    if algorithm_name == "MAPELITES":
        print("Start Running MAPELITES")
        mutation_power = algorithm_config["mutation_power"]
        initial_population = algorithm_config["initial_population"]
        algorithm_instance = MapElitesAlgorithm(mutation_power,
                                                initial_population,
                                                num_simulations,
                                                feature_map,)
    elif algorithm_name=="RANDOM":
        print("Start Running RANDOM")
        algorithm_instance=RandomGenerator(num_simulations,
                                           feature_map,)

    # create loggers
    individual_log = RunningIndividualLog("individuals_log.csv",
                                          elite_map_config)
    elite_map_log = FrequentMapLog("elite_map.csv",
                                   len(elite_map_config["Map"]["Features"]))
    map_summary_log = MapSummaryLog("map_summary.csv")

    # run search
    start_time = time.time()
    simulation = 1
    init_workers(visualize, elite_map_config, num_cores, model_path)
    while algorithm_instance.is_running():
        # print("Looking for idle workers")

        # assign job to idle workers
        while len(idle_workers) > 0 and simulation <= num_simulations:
            ind = algorithm_instance.generate_individual()
            print("Starting simulation: %d/%d on worker %d"
                % (simulation, num_simulations, idle_workers[0]))

            worker_id = idle_workers.pop(0)
            assign_eval_task(ind, worker_id, simulation)
            running_workers.append(worker_id)

            simulation += 1

        # find done workers
        num_running_workers = len(running_workers)
        for _ in range(num_running_workers):
            worker_id = running_workers.pop(0)
            if worker_has_finished(worker_id):
                # receive done individual
                worker = worker_list[worker_id]
                evaluated_ind = worker.get_ind()
                algorithm_instance.return_evaluated_individual(evaluated_ind)

                # log result
                individual_log.log_individual(evaluated_ind)
                elite_map_log.log_map(algorithm_instance.feature_map)
                map_summary_log.log_summary(algorithm_instance.feature_map,
                                            algorithm_instance.individuals_evaluated)

                # deal with workers
                idle_workers.append(worker_id)
                print("""Finished simulation: %d\nTotal simulation done: %d/%d"""
                      % (worker.get_sim_id(),
                         algorithm_instance.individuals_evaluated,
                         num_simulations))
            else:
                running_workers.append(worker_id)
        time.sleep(1)

    finish_time = time.time()
    print("Total evaluation time: " + str(finish_time - start_time) + " seconds")
    terminate_all_workers(num_cores)


def run(config,
        model_path,):
    """
    Read in toml config files and run the search

    Args:
        config (toml): toml config object of current experiment
        model_path (string): file path to the GAN model
    """
    # read in necessary config files
    experiment_config = toml.load(config)
    visualize = experiment_config["visualize"]
    num_cores = experiment_config["num_cores"]
    num_simulations = experiment_config["num_simulations"]
    algorithm_config = toml.load(
        os.path.join(LSI_CONFIG_ALGO_DIR,
        experiment_config["algorithm_config"]))
    elite_map_config = toml.load(
        os.path.join(LSI_CONFIG_MAP_DIR,
        experiment_config["elite_map_config"]))

    # run lsi search
    search(num_simulations,
           algorithm_config,
           elite_map_config,
           model_path,
           visualize,
           num_cores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', help='path of experiment config file',
                        required=False, default=os.path.join(LSI_CONFIG_EXP_DIR, "MAPELITES_demo.tml"))
    parser.add_argument('-m', '--model_path', help='path of the GAN trained',
                        required=False, default=os.path.join(GAN_TRAINING_DIR, "netG_epoch_49999_999.pth"))
    opt = parser.parse_args()
    run(opt.config,
        opt.model_path)