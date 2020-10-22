"""Runs a search to illuminate the latent space."""
import argparse
import os
import time

import ray
import toml
import torch
from overcooked_ai_pcg import (GAN_TRAINING_DIR, LSI_CONFIG_ALGO_DIR,
                               LSI_CONFIG_EXP_DIR, LSI_CONFIG_MAP_DIR)
from overcooked_ai_pcg.helper import read_gan_param, read_in_lsi_config
from overcooked_ai_pcg.LSI.evaluator import EvaluationActor
from overcooked_ai_pcg.LSI.logger import (FrequentMapLog, MapSummaryLog,
                                          RunningIndividualLog)
from overcooked_ai_pcg.LSI.qd_algorithms import (FeatureMap,
                                                 MapElitesAlgorithm,
                                                 RandomGenerator)


def search(num_simulations, algorithm_config, elite_map_config, model_path,
           visualize, num_cores):
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
        feature_ranges.append((bc["low"], bc["high"]))
        resolutions.append(bc["resolution"])
    feature_map = FeatureMap(num_simulations, feature_ranges, resolutions)

    # create loggers
    running_individual_log = RunningIndividualLog("individuals_log.csv",
                                                  elite_map_config)
    frequent_map_log = FrequentMapLog("elite_map.csv",
                                      len(elite_map_config["Map"]["Features"]))
    map_summary_log = MapSummaryLog("map_summary.csv")

    # configuration for running an actor on the head node
    node_id = f"node:{ray.services.get_node_ip_address()}"
    head_node_options = {
        "num_cpus": 1,
        "resources": {
            node_id: 0.01
        },
    }

    # config algorithm instance -> it runs on the head node so that it can
    # access the logger files
    algorithm_name = algorithm_config["name"]
    if algorithm_name == "MAPELITES":
        print("Start Running MAPELITES")
        mutation_power = algorithm_config["mutation_power"]
        initial_population = algorithm_config["initial_population"]
        # pylint: disable=no-member
        algorithm_instance = MapElitesAlgorithm.options(
            **head_node_options).remote(
                mutation_power,
                initial_population,
                num_simulations,
                feature_map,
                running_individual_log,
                frequent_map_log,
                map_summary_log,
            )
    elif algorithm_name == "RANDOM":
        print("Start Running RANDOM")
        # pylint: disable=no-member
        algorithm_instance = RandomGenerator.options(
            **head_node_options).remote(
                num_simulations,
                feature_map,
                running_individual_log,
                frequent_map_log,
                map_summary_log,
            )

    # run search
    start_time = time.time()

    G_params = read_gan_param()
    gan_state_dict = torch.load(model_path,
                                map_location=lambda storage, loc: storage)

    # each task repeatedly evaluates individuals and returns the evaluations to
    # the algorithm
    # pylint: disable=no-member
    evaluation_actors = [EvaluationActor.remote() for _ in range(num_cores)]
    handles = [
        actor.run_evaluation_loop.remote(algorithm_instance, visualize,
                                         elite_map_config, gan_state_dict,
                                         G_params, num_simulations, worker_id,
                                         evaluation_actors)
        for worker_id, actor in enumerate(evaluation_actors)
    ]

    # wait for the tasks to complete (i.e. for the algorithm to finish)
    try:
        ray.get(handles)
    except ray.exceptions.RayActorError:
        # this is expected to happen because the actors all get killed when one
        # of them finishes
        finish_time = time.time()
        print("Total evaluation time:", str(finish_time - start_time),
              "seconds")


def run(
    config,
    model_path,
):
    """
    Read in toml config files and run the search

    Args:
        config (toml): toml config object of current experiment
        model_path (string): file path to the GAN model
    """
    # read in necessary config files
    experiment_config, algorithm_config, elite_map_config = read_in_lsi_config(
        opt.config)

    visualize = experiment_config["visualize"]
    num_cores = experiment_config["num_cores"]
    num_simulations = experiment_config["num_simulations"]

    # initialize Ray
    ray.init(address="auto")

    # run lsi search
    search(num_simulations, algorithm_config, elite_map_config, model_path,
           visualize, num_cores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        help='path of experiment config file',
                        required=False,
                        default=os.path.join(LSI_CONFIG_EXP_DIR,
                                             "MAPELITES_demo.tml"))
    parser.add_argument('-m',
                        '--model_path',
                        help='path of the GAN trained',
                        required=False,
                        default=os.path.join(GAN_TRAINING_DIR,
                                             "netG_epoch_49999_999.pth"))
    opt = parser.parse_args()
    run(opt.config, opt.model_path)
