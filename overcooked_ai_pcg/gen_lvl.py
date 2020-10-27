import json
import os
import subprocess
import time
import argparse

import numpy as np
import torch
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from torch.autograd import Variable

from overcooked_ai_pcg import GAN_TRAINING_DIR, LSI_CONFIG_EXP_DIR
from overcooked_ai_pcg.GAN_training import dcgan
from overcooked_ai_pcg.helper import (gen_int_rnd_lvl, lvl_number2str,
                                      obj_types, read_gan_param,
                                      run_overcooked_game, setup_env_from_grid,
                                      read_in_lsi_config, lvl_str2grid)
from overcooked_ai_pcg.milp_repair import repair_lvl
from overcooked_ai_pcg.LSI.qd_algorithms import Individual

def generate_lvl(batch_size, generator, latent_vector=None, worker_id=0):
    """
    Generate level string from random latent vector given the path to the train netG model, and use MILP solver to repair it

    Args:
        generator (DCGAN): netG model
        latent_vector: np.ndarray with the required dimension.
                       When it is None, a new vector will be randomly sampled
    """
    # read in G constructor params from file
    # G_params = read_gan_param()
    nz = generator.nz
    x = np.random.randn(batch_size, nz, 1, 1)

    # generator = dcgan.DCGAN_G(**G_params)
    # generator.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    if latent_vector is None:
        latent_vector = torch.FloatTensor(x).view(batch_size, nz, 1, 1)
    else:
        latent_vector = torch.FloatTensor(latent_vector).view(
            batch_size, nz, 1, 1)
    with torch.no_grad():
        levels = generator(Variable(latent_vector))
    levels.data = levels.data[:, :, :10, :15]
    im = levels.data.cpu().numpy()
    im = np.argmax(im, axis=1)
    lvl_int = im[0]

    print("worker(%d): Before repair:\n" % (worker_id) +
          lvl_number2str(lvl_int))

    # In order to avoid dealing with memory leaks that may arise with docplex,
    # we run `repair_lvl` in a separate process. We can't create a child process
    # since the Dask workers are daemonic processes (which cannot spawn
    # children), so we run with subprocess. The code below essentially calls
    # `python` with a small bit of code and gets back the repr of a numpy array.
    # We then eval that output to get the repaird level.
    #
    # Yes, this is sketchy.

    # Allows us to separate the array from the rest of the process's output.
    delimiter = "----DELIMITER----DELIMITER----"
    output = subprocess.run(
        [
            'python', '-c', f"""\
import numpy as np
from numpy import array
from overcooked_ai_pcg.milp_repair import repair_lvl
np_lvl = eval(\"\"\"{np.array_repr(lvl_int)}\"\"\")
repaired_lvl = np.array_repr(repair_lvl(np_lvl))
print("{delimiter}")
print(repaired_lvl)
"""
        ],
        stdout=subprocess.PIPE,
    ).stdout.decode('utf-8')
    # Array comes after the delimiter.
    output = output.split(delimiter)[1]
    # The repr uses array and uint8 without np, so we make it available for eval
    # here.
    array, uint8 = np.array, np.uint8  # pylint: disable = unused-variable
    # Get the array.
    lvl_repaired = eval(output)

    lvl_str = lvl_number2str(lvl_repaired)

    print("worker(%d): After repair:\n" % (worker_id) + lvl_str)
    return lvl_str


def generate_rnd_lvl(size, worker_id=0):
    """
    generate random level of specified size and fix it using MILP solver

    Args:
        size: 2D tuple of integers with format (height, width)
    """
    rnd_lvl_int = gen_int_rnd_lvl(size)

    print("worker(%d): Before repair:\n" % (worker_id) +
          lvl_number2str(rnd_lvl_int))

    # print("Start MILP repair...")
    lvl_repaired = repair_lvl(rnd_lvl_int)
    lvl_str = lvl_number2str(lvl_repaired)

    print("worker(%d): After repair:\n" % (worker_id) + lvl_str)
    return lvl_str


def main(config):
    _, _, _, agent_config = read_in_lsi_config(config)
    G_params = read_gan_param()
    gan_state_dict = torch.load(os.path.join(GAN_TRAINING_DIR,
                                             "netG_epoch_49999_999.pth"),
                                map_location=lambda storage, loc: storage)
    generator = dcgan.DCGAN_G(**G_params)
    generator.load_state_dict(gan_state_dict)
    lvl_str = generate_lvl(1, generator)

    # lvl_str = """XXPXX
    #              T  2T
    #              X1  O
    #              XXDSX
    #              """
    # lvl_str = """XXXPPXXX
    #              X  2   X
    #              D XXXX S
    #              X  1   X
    #              XXXOOXXX
    #              """
    # lvl_str = """XXXXXXXXXXXXXXX
    #              O 1     XX    D
    #              X  X2XXXXXXXX S
    #              O XX     XXXX X
    #              X         X   X
    #              O          X  X
    #              X  XXXXXXX    X
    #              X  XXXX  PXXX X
    #              X          X  X
    #              XXXXXXXXXXXXXXX
    #              """
    # lvl_str = generate_rnd_lvl((5, 5))

    ind = Individual()
    fitness, _, _, _ = run_overcooked_game(ind, lvl_str, agent_config, render=True)
    print("fitness: %d" % fitness)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        help='path of experiment config file',
                        required=False,
                        default=os.path.join(LSI_CONFIG_EXP_DIR,
                                             "MAPELITES_workloads_diff_fixed_plan.tml"))
    opt = parser.parse_args()
    main(opt.config)
