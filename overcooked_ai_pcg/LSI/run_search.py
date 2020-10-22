"""Runs a search to illuminate the latent space."""
import argparse
import os
import time

import toml
import torch
from dask.distributed import Client, LocalCluster, as_completed
from dask_jobqueue import SLURMCluster
from overcooked_ai_pcg import (GAN_TRAINING_DIR, LSI_CONFIG_ALGO_DIR,
                               LSI_CONFIG_EXP_DIR, LSI_CONFIG_MAP_DIR)
from overcooked_ai_pcg.helper import read_gan_param
from overcooked_ai_pcg.LSI.evaluator import run_overcooked_eval
from overcooked_ai_pcg.LSI.logger import (FrequentMapLog, MapSummaryLog,
                                          RunningIndividualLog)
from overcooked_ai_pcg.LSI.qd_algorithms import (FeatureMap,
                                                 MapElitesAlgorithm,
                                                 RandomGenerator)


def search(dask_client, num_simulations, algorithm_config, elite_map_config,
           model_path, visualize, num_cores):
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

    # config algorithm instance -> it runs on the head node so that it can
    # access the logger files
    algorithm_name = algorithm_config["name"]
    if algorithm_name == "MAPELITES":
        print("Start Running MAPELITES")
        mutation_power = algorithm_config["mutation_power"]
        initial_population = algorithm_config["initial_population"]
        # pylint: disable=no-member
        algorithm_instance = MapElitesAlgorithm(
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
        algorithm_instance = RandomGenerator(
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

    # initialize the workers with num_cores + 1 jobs (the additional one should
    # keep the workers busy while this main thread processes completed
    # individuals)
    evaluations = []
    for worker_id in range(1, num_cores + 2):
        evaluations.append(
            dask_client.submit(
                run_overcooked_eval,
                algorithm_instance.generate_individual(),
                visualize,
                elite_map_config,
                G_params,
                gan_state_dict,
                worker_id,
            ))
    as_completed_evaluations = as_completed(evaluations)
    print(f"Started {num_cores} simulations")

    # repeatedly grab completed evaluations, return them to the algorithm, and
    # send out new evaluations
    for completion in as_completed_evaluations:
        evaluated_ind = completion.result()

        # evaluated_ind may be None if the evaluation failed
        if (evaluated_ind is not None and
                algorithm_instance.insert_if_still_running(evaluated_ind)):
            print("Finished simulation.\nTotal simulations done: %d/%d" %
                  (algorithm_instance.individuals_evaluated, num_simulations))

        if algorithm_instance.is_running():
            # request another evaluation if still running
            new_ind = algorithm_instance.generate_individual()
            future = dask_client.submit(
                run_overcooked_eval,
                new_ind,
                visualize,
                elite_map_config,
                G_params,
                gan_state_dict,
                # since there are no more "workers", we just pass in the
                # id of the individual as the worker id
                algorithm_instance.individuals_disbatched,
            )
            as_completed_evaluations.add(future)
            evaluations.append(future)
            print("Starting simulation: %d/%d" %
                  (algorithm_instance.individuals_disbatched, num_simulations))
        else:
            # otherwise, terminate
            break

    dask_client.cancel(evaluations)  # Cancel remaining evals.
    finish_time = time.time()
    print("Total evaluation time:", str(finish_time - start_time), "seconds")


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
    experiment_config = toml.load(config)
    visualize = experiment_config["visualize"]
    num_cores = experiment_config["num_cores"]
    num_simulations = experiment_config["num_simulations"]
    algorithm_config = toml.load(
        os.path.join(LSI_CONFIG_ALGO_DIR,
                     experiment_config["algorithm_config"]))
    elite_map_config = toml.load(
        os.path.join(LSI_CONFIG_MAP_DIR, experiment_config["elite_map_config"]))

    # Initialize Dask.
    if experiment_config.get("slurm", False):
        cores_per_worker = experiment_config["num_cores_per_slurm_worker"]

        # 1 process per CPU since cores == processes
        cluster = SLURMCluster(
            project=experiment_config["slurm_project"],
            cores=cores_per_worker,
            memory=f"{experiment_config['mem_gib_per_slurm_worker']}GiB",
            processes=cores_per_worker,
        )

        print("### SLURM Job script ###")
        print("--------------------------------------")
        print(cluster.job_script())
        print("--------------------------------------")

        cluster.scale(cores=num_cores)
        dask_client = Client(cluster)
    else:
        # Single machine -- run with num_cores worker processes.
        cluster = LocalCluster(n_workers=num_cores,
                               threads_per_worker=1,
                               processes=True)
        dask_client = Client(cluster)

    # run lsi search
    search(dask_client, num_simulations, algorithm_config, elite_map_config,
           model_path, visualize, num_cores)


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
