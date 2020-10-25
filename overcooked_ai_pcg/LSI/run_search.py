"""Runs a search to illuminate the latent space."""
import argparse
import os
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
from overcooked_ai_pcg.LSI.qd_algorithms import (FeatureMap,
                                                 MapElitesAlgorithm,
                                                 RandomGenerator)


def init_logging_dir(config_path, experiment_config, algorithm_config,
                     elite_map_config):
    """Creates the logging directory, saves configs to it, and starts a README.

    Args:
        config_path (str): path to the experiment config file
        experiment_config (toml): toml config object of current experiment
        algorithm_config: toml config object of QD algorithm
        elite_map_config: toml config object of the feature maps
    Returns:
        log_dir: full path to the logging directory
        base_log_dir: the path without LSI_LOG_DIR prepended
    """
    # create logging directory
    exp_name = os.path.basename(config_path).replace(".tml",
                                                     "").replace("_", "-")
    time_str = time.strftime("%Y-%m-%d_%k-%M-%S")
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
            },
            file,
        )

    # start a README
    with open(os.path.join(log_dir, "README.md"), "w") as file:
        file.write(f"# {exp_name}, {time_str}\n")

    return log_dir, base_log_dir


def init_dask(experiment_config, log_dir):
    """Initializes Dask with a local or SLURM cluster.

    Args:
        experiment_config (toml): toml config object of experiment
        log_dir (str): directory for storing logs
    Returns:
        A Dask client for the cluster created.
    """
    num_cores = experiment_config["num_cores"]

    if experiment_config.get("slurm", False):
        worker_logs = os.path.join(log_dir, "worker_logs")
        os.mkdir(worker_logs)

        cores_per_worker = experiment_config["num_cores_per_slurm_worker"]

        # 1 process per CPU since cores == processes
        cluster = SLURMCluster(
            project=experiment_config["slurm_project"],
            cores=cores_per_worker,
            memory=f"{experiment_config['mem_gib_per_slurm_worker']}GiB",
            processes=cores_per_worker,
            walltime=experiment_config['slurm_worker_walltime'],
            job_extra=[
                f"--output {worker_logs}/slurm-%j.out",
                f"--error {worker_logs}/slurm-%j.out",
            ],
        )

        print("### SLURM Job script ###")
        print("--------------------------------------")
        print(cluster.job_script())
        print("--------------------------------------")

        cluster.scale(cores=num_cores)
        return dask.distributed.Client(cluster)

    # Single machine -- run with num_cores worker processes.
    cluster = dask.distributed.LocalCluster(n_workers=num_cores,
                                            threads_per_worker=1,
                                            processes=True)
    return dask.distributed.Client(cluster)


def search(dask_client, base_log_dir, num_simulations, algorithm_config,
           elite_map_config, model_path, visualize, num_cores):
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
    running_individual_log = RunningIndividualLog(
        os.path.join(base_log_dir, "individuals_log.csv"),
        elite_map_config,
    )
    frequent_map_log = FrequentMapLog(
        os.path.join(base_log_dir, "elite_map.csv"),
        len(elite_map_config["Map"]["Features"]),
    )
    map_summary_log = MapSummaryLog(
        os.path.join(base_log_dir, "map_summary.csv"))

    # config algorithm instance -> it runs on the head node so that it can
    # access the logger files
    algorithm_name = algorithm_config["name"]
    if algorithm_name == "MAPELITES":
        print("Start Running MAPELITES")
        mutation_power = algorithm_config["mutation_power"]
        initial_population = algorithm_config["initial_population"]
        # pylint: disable=no-member
        algorithm = MapElitesAlgorithm(
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
        algorithm = RandomGenerator(
            num_simulations,
            feature_map,
            running_individual_log,
            frequent_map_log,
            map_summary_log,
        )

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
        evaluations.append(
            dask_client.submit(
                run_overcooked_eval,
                algorithm.generate_individual(),
                visualize,
                elite_map_config,
                G_params,
                gan_state_dict,
                active_evals + 1,  # worker_id
            ))
        active_evals += 1
    evaluations = dask.distributed.as_completed(evaluations)
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
            elif (evaluated_ind is not None and
                  algorithm.insert_if_still_running(evaluated_ind)):
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
                    G_params,
                    gan_state_dict,
                    # since there are no more "workers", we just pass in the
                    # id of the individual as the worker id
                    algorithm.individuals_disbatched,
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
):
    """
    Read in toml config files and run the search

    Args:
        config (toml): toml config path of current experiment
        model_path (string): file path to the GAN model
    """
    experiment_config, algorithm_config, elite_map_config = \
        read_in_lsi_config(config)

    log_dir, base_log_dir = init_logging_dir(config, experiment_config,
                                             algorithm_config, elite_map_config)
    print("LOGGING DIRECTORY:", log_dir)

    # start LSI search
    search(
        init_dask(experiment_config, log_dir),
        base_log_dir,
        experiment_config["num_simulations"],
        algorithm_config,
        elite_map_config,
        model_path,
        experiment_config["visualize"],
        experiment_config["num_cores"],
    )


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
