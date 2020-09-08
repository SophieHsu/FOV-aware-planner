import os
import toml
import argparse
from overcooked_ai_pcg.LSI.search import *
from overcooked_ai_pcg import GAN_TRAINING_DIR, LSI_CONFIG_EXP_DIR, LSI_CONFIG_ALGO_DIR, LSI_CONFIG_MAP_DIR

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