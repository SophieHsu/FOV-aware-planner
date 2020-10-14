import os
import numpy as np
import time
import torch
import json
from torch.autograd import Variable
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_pcg import GAN_TRAINING_DIR
from overcooked_ai_pcg.GAN_training import dcgan
from overcooked_ai_pcg.milp_repair import repair_lvl
from overcooked_ai_pcg.helper import obj_types, lvl_number2str, setup_env_from_grid, read_gan_param, run_overcooked_game, gen_int_rnd_lvl


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

    # print("Start MILP repair...")
    lvl_repaired = repair_lvl(lvl_int)
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


def main():
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

    grid = [layout_row.strip() for layout_row in lvl_str.split("\n")][:-1]

    reward, _ = run_overcooked_game(lvl_str, render=True)
    print("reward: %d" % reward)


if __name__ == "__main__":
    main()