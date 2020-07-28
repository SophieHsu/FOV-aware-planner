import numpy as np
import os
import json
import torch
from matplotlib import pyplot as plt
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import *
from overcooked_ai_py import read_layout_dict
from overcooked_ai_py import LAYOUTS_DIR
from overcooked_ai_pcg import ERR_LOG_PIC, G_PARAM_FILE

obj_types = "12XSPOD "

def vertical_flip(np_lvl):
    """
    Return the vertically flipped version of the input np level.
    """
    np_lvl_vflip = np.zeros(np_lvl.shape)
    height, width = np_lvl.shape
    for x in range(height):
        for y in range(width):
            np_lvl_vflip[x][y] = np_lvl[x][width-y-1]
    
    return np_lvl_vflip.astype(np.uint8)

def horizontal_flip(np_lvl):
    """
    Return the horizontally flipped version of the input np level.
    """
    np_lvl_hflip = np.zeros(np_lvl.shape)
    height = np_lvl.shape[0]
    for x in range(height):
        np_lvl_hflip[x] = np_lvl[height-x-1]
    return np_lvl_hflip.astype(np.uint8)

def lvl_str2number(raw_layout):
    """
    Turns pure string formatted lvl to num encoded format
    """
    np_lvl = np.zeros((len(raw_layout), len(raw_layout[0])))
    for x, row in enumerate(raw_layout):
        row = row.strip()
        for y, tile in enumerate(row):
            np_lvl[x,y] = obj_types.index(tile)
    return np_lvl

def lvl_number2str(np_lvl):
    """
    Turns num encoded format to pure string formatted lvl 
    """
    lvl_str = ""
    for lvl_row in np_lvl:
        for tile_int in lvl_row:
            lvl_str += obj_types[tile_int]
        lvl_str += "\n"
    return lvl_str

def read_in_training_data(data_path):
    """
    Read in .layouts file and return the data

    Args:
        data_path: path to the directory containing the training data

    returns: a 3D np array of size num_lvl x lvl_height x lvl_width 
             containing the encoded levels
    """
    lvls = []
    for layout_file in os.listdir(data_path):
        if layout_file.endswith(".layout") and layout_file.startswith("gen"):
            layout_name = layout_file.split('.')[0]
            raw_layout = read_layout_dict(layout_name)
            raw_layout = raw_layout['grid'].split('\n')

            np_lvl = lvl_str2number(raw_layout)
            
            # data agumentation: add flipped levels to data set
            np_lvl = np_lvl.astype(np.uint8)
            np_lvl_vflip = vertical_flip(np_lvl)
            np_lvl_hflip = horizontal_flip(np_lvl)
            np_lvl_vhflip = vertical_flip(np_lvl_hflip)
            lvls.append(np_lvl)
            lvls.append(np_lvl_vflip)
            lvls.append(np_lvl_hflip)
            lvls.append(np_lvl_vhflip)

    return np.array(lvls)

# print(read_in_training_data(LAYOUTS_DIR))

def plot_err(average_errG_log, 
             average_errD_log, 
             average_errD_fake_log, 
             average_errD_real_log,
             average_D_x_log,
             average_D_G_z1_log,
             average_D_G_z2_log):
    """
    Given lists of recorded errors and plot them.
    """
    plt.subplot(2, 2, 1)
    plt.plot(average_errD_log, 'b', label="err_D")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(average_errD_fake_log, 'r', label="err_D_fake")
    plt.plot(average_errD_real_log, 'g', label="err_D_real")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(average_errG_log, 'r', label="err_G")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(average_D_x_log, 'r', label="D(x)")
    plt.plot(average_D_G_z1_log, 'g', label="D(G(z1))")
    plt.plot(average_D_G_z2_log, 'b', label="D(G(z2))")
    plt.legend()

    plt.savefig(ERR_LOG_PIC)
    plt.show()

def setup_env_from_grid(layout_grid, config):
    """
    Set up random agents and overcooked env to run demo game.

    Args:
        layout_grid: list of string each representing a row of layout
        config: metadata to config the env
    """
    mdp = OvercookedGridworld.from_grid(layout_grid, config)
    env = OvercookedEnv.from_mdp(mdp, info_level = 0)
    agent1 = RandomAgent(all_actions=True)
    agent2 = RandomAgent(all_actions=True)
    agent1.set_agent_index(0)
    agent2.set_agent_index(1)
    agent1.set_mdp(mdp)
    agent2.set_mdp(mdp)
    return agent1, agent2, env

def save_gan_param(G_params):
    with open(G_PARAM_FILE, "w") as f:
        json.dump(G_params, f)

def read_gan_param():
    with open(G_PARAM_FILE, "r") as f:
        G_params = json.load(f)
    return G_params