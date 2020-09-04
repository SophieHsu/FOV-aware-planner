import os
import toml
import argparse
from overcooked_ai_pcg.LSI.search import *
from overcooked_ai_pcg import GAN_TRAINING_DIR, LSI_CONFIG_EXP_DIR

def run(config,
        workerID,
        model_path,):
    experiment_toml = toml.load(config)
    visualize = experiment_toml["Visualize"]
    num_cores = experiment_toml["num_cores"]
    num_list = []
    workerID = workerID-1
    if workerID < 0:
        print("workerID should be greater than 0")
    else:
        for i in experiment_toml["Trials"]:
            num_list.append(i["num_trials"])

        # determine which trial to run
        for trial_index in range(len(num_list)):
            if workerID < num_list[trial_index]:
                start_search(workerID+1, trial_index, experiment_toml, model_path, visualize, num_cores)
                print("workerID " + str(workerID) + " finished")
                break
            workerID = workerID - num_list[trial_index]
        if trial_index == len(num_list):
            print("workerID is greater than the total number of trials")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--workerID',help='the workerID of the worker to be called',type=int,
                        required=True)
    parser.add_argument('-c','--config', help='path of experiment config file',
                        required=False, default=os.path.join(LSI_CONFIG_EXP_DIR, "experiments.tml"))
    parser.add_argument('-m', '--model_path', help='path of the GAN trained',
                        required=False, default=os.path.join(GAN_TRAINING_DIR, "netG_epoch_49999_999.pth"))
    opt = parser.parse_args()
    run(opt.config,
        opt.workerID,
        opt.model_path)