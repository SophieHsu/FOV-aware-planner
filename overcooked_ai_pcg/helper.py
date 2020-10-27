import os
import json
import torch
import time
import toml
import numpy as np
from matplotlib import pyplot as plt
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner, MediumLevelMdpPlanner, HumanMediumLevelPlanner, HumanAwareMediumMDPPlanner, MediumLevelActionManager
from overcooked_ai_py.agents.agent import *
from overcooked_ai_py.planning.planners import Heuristic
from overcooked_ai_py import read_layout_dict
from overcooked_ai_py import LAYOUTS_DIR
from overcooked_ai_pcg import ERR_LOG_PIC, G_PARAM_FILE, LSI_CONFIG_ALGO_DIR, LSI_CONFIG_MAP_DIR, LSI_CONFIG_AGENT_DIR

import gc

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

def lvl_str2grid(lvl_str):
    """
    Turns pure string formatted lvl to grid format compatible with overcooked-AI env
    """
    return [layout_row.strip() for layout_row in lvl_str.split("\n")][:-1]

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

def read_in_lsi_config(exp_config_file):
    experiment_config = toml.load(exp_config_file)
    algorithm_config = toml.load(
        os.path.join(LSI_CONFIG_ALGO_DIR,
                    experiment_config["algorithm_config"]))
    elite_map_config = toml.load(
        os.path.join(LSI_CONFIG_MAP_DIR, experiment_config["elite_map_config"]))
    agent_config = toml.load(
        os.path.join(LSI_CONFIG_AGENT_DIR, experiment_config["agent_config"]))
    return experiment_config, algorithm_config, elite_map_config, agent_config

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

def setup_env_from_grid(layout_grid, agent_config, worker_id=0, human_preference=0.3, human_adaptiveness=0.5):
    """
    Set up random agents and overcooked env to run demo game.

    Args:
        layout_grid: list of string each representing a row of layout
    """
    config = {
        "start_order_list": ['onion'] * 2,
        "cook_time": 10,
        "num_items_for_soup": 3,
        "delivery_reward": 20,
        "rew_shaping_params": None
    }
    mdp = OvercookedGridworld.from_grid(layout_grid, config)
    env = OvercookedEnv.from_mdp(mdp, info_level = 0, horizon = 100)

    base_params = {
        'start_orientations': False,
        'wait_allowed': False,
        'counter_goals': [],
        'counter_drop': [],
        'counter_pickup': [],
        'same_motion_goals': True
    }
    start_time = time.time()
    
    agent1_config = agent_config["Agent1"]
    agent2_config = agent_config["Agent2"]

    # # Set up 1: two coupled planning agent
    # mlp_planner1 = MediumLevelPlanner(mdp, base_params)
    # mlp_planner2 = MediumLevelPlanner(mdp, base_params)

    # agent1 = CoupledPlanningAgent(mlp_planner1)
    # agent2 = CoupledPlanningAgent(mlp_planner2)

    # # Set up 2: Stayagent + GreedyHumanModel
    # mlp_planner = MediumLevelPlanner(mdp, base_params)
    # agent1 = StayAgent()
    # agent2 = GreedyHumanModel(mlp_planner, env)

    # Set up 3: Fixed plan agents
    if agent1_config["name"] == "fixed_plan_agent" and agent2_config["name"] == "fixed_plan_agent":
        print("worker(%d): Pre-constructing graph..." % (worker_id))
        mlp_planner = MediumLevelPlanner(mdp, base_params)
        print("worker(%d): Planning..." % (worker_id))
        joint_plan = \
            mlp_planner.get_low_level_action_plan(
                env.state,
                Heuristic(mlp_planner.mp).simple_heuristic,
                delivery_horizon=agent_config["joint_plan"]["delivery_horizon"],
                goal_info=agent_config["joint_plan"]["goal_info"])

        plan1 = []
        plan2 = []
        for joint_action in joint_plan:
            action1, action2 = joint_action
            plan1.append(action1)
            plan2.append(action2)

        agent1 = FixedPlanAgent(plan1)
        agent2 = FixedPlanAgent(plan2)
        del mlp_planner

    # Set up 4: Preferenced human + human-aware agent
    elif agent1_config["name"] == "preferenced_human" and agent2_config["name"] == "human_aware_agent":
        print("worker(%d): Pre-constructing graph..." % (worker_id))
        ml_action_manager = MediumLevelActionManager(mdp, base_params)
        hmlp = HumanMediumLevelPlanner(mdp, ml_action_manager, [human_preference, 1.0-human_preference], human_adaptiveness)
        print("worker(%d): Planning..." % (worker_id))

        agent1 = biasHumanModel(ml_action_manager, [human_preference, 1.0-human_preference], human_adaptiveness, auto_unstuck=agent1_config["auto_unstuck"])
        # agent1 = oneGoalHumanModel(mlp_planner, 'Onion cooker', auto_unstuck=True)
        print("worker(%d): Pre-constructing mdp plan..." % (worker_id))
        
        mdp_planner = HumanAwareMediumMDPPlanner.from_pickle_or_compute(mdp, base_params, hmlp, ml_action_manager, force_compute_all=True)
        print("worker(%d): MDP agent planning..." % (worker_id))
        
        agent2 = MediumMdpPlanningAgent(mdp_planner, env, auto_unstuck=agent2_config["auto_unstuck"])


        print("worker(%d): Preprocess take %d seconds"
            % (worker_id, time.time() - start_time))
        agent1.set_agent_index(0)
        agent2.set_agent_index(1)
        agent1.set_mdp(mdp)
        agent2.set_mdp(mdp)

        del ml_action_manager, hmlp, mdp_planner
    gc.collect()

    return agent1, agent2, env

def save_gan_param(G_params):
    with open(G_PARAM_FILE, "w") as f:
        json.dump(G_params, f)

def read_gan_param():
    with open(G_PARAM_FILE, "r") as f:
        G_params = json.load(f)
    return G_params

def run_overcooked_game(ind, lvl_str, agent_config, render=True, worker_id=0):
    """
    Run one turn of overcooked game and return the sparse reward as fitness
    """
    grid = lvl_str2grid(lvl_str)
    agent1, agent2, env = setup_env_from_grid(grid, agent_config, worker_id=worker_id, human_preference=ind.human_preference, human_adaptiveness=ind.human_adaptiveness)
    done = False
    total_sparse_reward = 0
    last_state = None
    timestep = 0

    # Saves when each soup (order) was delivered
    checkpoints = [env.horizon-1] * env.num_orders
    cur_order = 0

    while not done:
        if render:
            env.render()
            time.sleep(0.5)
        # print("start compute actions")
        joint_action = (agent1.action(env.state)[0], agent2.action(env.state)[0])
        # print(joint_action)
        next_state, timestep_sparse_reward, done, info = env.step(joint_action)
        total_sparse_reward += timestep_sparse_reward
        
        if timestep_sparse_reward > 0:
            checkpoints[cur_order] = timestep
            cur_order += 1
        
        last_state = next_state
        timestep += 1

    workloads = last_state.get_player_workload()

    # Smooth fitness is the total reward tie-broken by soup delivery times.
    # Later soup deliveries are higher priority.
    fitness = total_sparse_reward
    for time in reversed(checkpoints):
        fitness *= env.horizon
        fitness += time

    # Free up some memory
    del agent1, agent2, env
    
    print("COMPLETE:", fitness, total_sparse_reward, checkpoints)
    return fitness, total_sparse_reward, checkpoints, workloads

def gen_int_rnd_lvl(size):
    """
    Randomly generate an unfixed integer level of specified size

    Args:
        size: 2D tuple of integers with format (height, width)
    """
    return np.random.randint(len(obj_types), size=size)
