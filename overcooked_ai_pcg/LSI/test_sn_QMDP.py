"""Runs a search to illuminate the latent space."""
import argparse
import os
import subprocess
import time

import dask.distributed
import toml
import torch
from dask_jobqueue import SLURMCluster

from overcooked_ai_pcg import GAN_TRAINING_DIR, LSI_CONFIG_EXP_DIR, LSI_LOG_DIR
from overcooked_ai_pcg.helper import read_gan_param, read_in_lsi_config
from overcooked_ai_pcg.LSI.evaluator import run_overcooked_eval
from overcooked_ai_pcg.LSI.logger import (FrequentMapLog, MapSummaryLog,
                                          RunningIndividualLog)
from overcooked_ai_pcg.LSI.qd_algorithms import (CMA_ME_Algorithm, FeatureMap,
                                                 MapElitesAlgorithm,
                                                 RandomGenerator,
                                                 MapElitesBaselineAlgorithm)


def init_logging_dir(config_path, experiment_config, algorithm_config,
                     elite_map_config, agent_configs):
    """Creates the logging directory, saves configs to it, and starts a README.

    Args:
        config_path (str): path to the experiment config file
        experiment_config (toml): toml config object of current experiment
        algorithm_config: toml config object of QD algorithm
        elite_map_config: toml config object of the feature maps
        agent_configs: toml config object of the agents used
    Returns:
        log_dir: full path to the logging directory
        base_log_dir: the path without LSI_LOG_DIR prepended
    """
    # create logging directory
    exp_name = os.path.basename(config_path).replace(".tml",
                                                     "").replace("_", "-")
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = time_str + "_" + exp_name
    log_dir = os.path.join(LSI_LOG_DIR, base_log_dir)
    os.mkdir(log_dir)

    # save configs
    with open(os.path.join(log_dir, "config.tml"), "w") as file:
        toml.dump(
            {
                "experiment_config": experiment_config,
                "algorithm_config": algorithm_config,
                "elite_map_config": elite_map_config,
                "agent_configs": agent_configs,
            },
            file,
        )

    # start a README
    with open(os.path.join(log_dir, "README.md"), "w") as file:
        file.write(f"# {exp_name}, {time_str}\n")

    return log_dir, base_log_dir



def search(base_log_dir, num_simulations, algorithm_config,
           elite_map_config, agent_configs, model_path, visualize, num_cores,
           lvl_size):
    """
    Run search with the specified algorithm and elite map

    Args:
        dask_client (dask.distributed.Client): client for accessing a Dask
            cluster.
        base_log_dir (str): Logging directory within LSI_LOG_DIR for storing
            logs.
        num_simulations (int): total number of evaluations of QD algorithm to run.
        algorithm_config: toml config object of QD algorithm
        elite_map_config: toml config object of the feature maps
        agent_configs (list): list of toml config object of agents
        model_path (string): file path to the GAN model
        visualize (bool): render the game or not
        num_cores (int): number of processes to run
        lvl_size (tuple): size of the level to generate. Currently only supports
                          (6, 9) and (10, 15)
    """

    # config feature map
    feature_ranges = []
    resolutions = []
    for bc in elite_map_config["Map"]["Features"]:
        feature_ranges.append((bc["low"], bc["high"]))
        resolutions.append(bc["resolution"])
    feature_map = FeatureMap(num_simulations, feature_ranges, resolutions)

    # create loggers
    running_individual_log = RunningIndividualLog(
        os.path.join(base_log_dir, "individuals_log.csv"), elite_map_config,
        agent_configs)
    frequent_map_log = FrequentMapLog(
        os.path.join(base_log_dir, "elite_map.csv"),
        len(elite_map_config["Map"]["Features"]),
    )
    map_summary_log = MapSummaryLog(
        os.path.join(base_log_dir, "map_summary.csv"))

    # config algorithm instance -> it runs on the head node so that it can
    # access the logger files
    algorithm_name = algorithm_config["name"]

    # take the max of all num params of all agent configs
    num_params = 0
    for agent_config in agent_configs:
        num_params = max(num_params, agent_config["Search"]["num_param"])
        print("Start CMA-ME")
        mutation_power = algorithm_config["mutation_power"]
        pop_size = algorithm_config["population_size"]
        algorithm = CMA_ME_Algorithm(mutation_power, num_simulations, pop_size,
                                     feature_map, running_individual_log,
                                     frequent_map_log, map_summary_log,
                                     num_params)

    # Super hacky! This is where we add bounded constraints for the human model.
    if num_params > 32:
        for i in range(32, num_params):
            algorithm.add_bound_constraint(i, (0.0, 1.0))

    # run search
    start_time = time.time()

    # GAN data
    G_params = read_gan_param()
     
    gan_state_dict = torch.load(model_path,
                                map_location=lambda storage, loc: storage)

    # initialize the workers with num_cores jobs
    evaluations = []
    active_evals = 0
    while active_evals < num_cores and not algorithm.is_blocking():
        run_overcooked_eval(algorithm.generate_individual(),
                visualize,
                elite_map_config,
                agent_configs,
                algorithm_config,
                G_params,
                gan_state_dict,
                active_evals + 1,  # worker_id
                lvl_size)

    
        # evaluations.append(
        #     dask_client.submit(
        #         run_overcooked_eval,
        #         algorithm.generate_individual(),
        #         visualize,
        #         elite_map_config,
        #         agent_configs,
        #         algorithm_config,
        #         G_params,
        #         gan_state_dict,
        #         active_evals + 1,  # worker_id
        #         lvl_size,
        #     ))
        active_evals += 1
    #evaluations = dask.distributed.as_completed(evaluations)
    print(f"Started {active_evals} simulations")

     
    # completion time of the latest simulation
    last_eval = time.time()

    # repeatedly grab completed evaluations, return them to the algorithm, and
    # send out new evaluations
    for completion in evaluations:
        # process the individual
        active_evals -= 1
        try:
            evaluated_ind = completion.result()

            if evaluated_ind is None:
                print("Received a failed evaluation.")
            elif (evaluated_ind is not None
                  and algorithm.insert_if_still_running(evaluated_ind)):
                cur_time = time.time()
                print("Finished simulation.\n"
                      f"Total simulations done: "
                      f"{algorithm.individuals_evaluated}/{num_simulations}\n"
                      f"Time since last simulation: {cur_time - last_eval}s\n"
                      f"Active evaluations: {active_evals}")
                last_eval = cur_time
        except dask.distributed.scheduler.KilledWorker as err:  # pylint: disable=no-member
            # worker may fail due to, for instance, memory
            print("Worker failed with the following error; continuing anyway\n"
                  "-------------------------------------------\n"
                  f"{err}\n"
                  "-------------------------------------------")
            continue

        del completion  # clean up

        if algorithm.is_running():
            # request more evaluations if still running
            while active_evals < num_cores and not algorithm.is_blocking():
                print("Starting simulation: ", end="")
                new_ind = algorithm.generate_individual()
                future = dask_client.submit(
                    run_overcooked_eval,
                    new_ind,
                    visualize,
                    elite_map_config,
                    agent_configs,
                    algorithm_config,
                    G_params,
                    gan_state_dict,
                    # since there are no more "workers", we just pass in the
                    # id of the individual as the worker id
                    algorithm.individuals_disbatched,
                    lvl_size,
                )
                evaluations.add(future)
                active_evals += 1
                print(f"{algorithm.individuals_disbatched}/{num_simulations}")
            print(f"Active evaluations: {active_evals}")
        else:
            # otherwise, terminate
            break

    finish_time = time.time()
    print("Total evaluation time:", str(finish_time - start_time), "seconds")


def run(
    config,
    model_path,
    lvl_size,
):
    """
    Read in toml config files and run the search

    Args:
        config (toml): toml config path of current experiment
        model_path (string): file path to the GAN model
    """
    experiment_config, algorithm_config, elite_map_config, agent_configs = \
        read_in_lsi_config(config)

    log_dir, base_log_dir = init_logging_dir(config, experiment_config,
                                             algorithm_config,
                                             elite_map_config, agent_configs)
    print("LOGGING DIRECTORY:", log_dir)

    # start LSI search
    search(
        base_log_dir,
        experiment_config["num_simulations"],
        algorithm_config,
        elite_map_config,
        agent_configs,
        model_path,
        experiment_config["visualize"],
        experiment_config["num_cores"],
        lvl_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        help='path of experiment config file',
                        required=False,
                        default=os.path.join(LSI_CONFIG_EXP_DIR,
                                             "MAPELITES_demo.tml"))
    parser.add_argument('-s',
                        '--size_version',
                        type=str,
                        default="small",
                        help='Size of the level. \
                             "small" for (6, 9), \
                             "large" for (10, 15)')
    # parser.add_argument('-m',
    #                     '--model_path',
    #                     help='path of the GAN trained',
    #                     required=False,
    #                     default=os.path.join(GAN_TRAINING_DIR,
    #                                          "netG_epoch_49999_999.pth"))
    opt = parser.parse_args()

    lvl_size = None
    gan_pth_path = None
    if opt.size_version == "small":
        lvl_size = (6, 9)
        gan_pth_path = os.path.join(GAN_TRAINING_DIR,
                                    "netG_epoch_49999_999_small.pth")
    elif opt.size_version == "large":
        lvl_size = (10, 15)
        gan_pth_path = os.path.join(GAN_TRAINING_DIR,
                                    "netG_epoch_49999_999_large.pth")
                                    
     
    #from IPython import embed
    #embed()
    run(opt.config, gan_pth_path, lvl_size)