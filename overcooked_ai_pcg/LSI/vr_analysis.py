import pygame
import os
import csv
import ast
import time
import toml
import shutil
import gc
import argparse
import random
import pickle
import numpy as np
import pandas as pd
from pprint import pprint
from overcooked_ai_pcg import (LSI_STEAK_STUDY_RESULT_DIR,
                               LSI_STEAK_STUDY_CONFIG_DIR,
                               LSI_STEAK_STUDY_AGENT_DIR)
from overcooked_ai_pcg.helper import init_steak_env, init_steak_qmdp_agent, lvl_str2grid, BASE_PARAMS
import overcooked_ai_py.agents.agent as agent
import overcooked_ai_py.planning.planners as planners
from overcooked_ai_py.agents.agent import HumanPlayer, StayAgent
from overcooked_ai_py.mdp.overcooked_mdp import Direction, Action, SteakHouseGridworld
from overcooked_ai_py.mdp.overcooked_env import MAX_HORIZON, OvercookedEnv

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

HUMAN_STUDY_ENV_HORIZON = 400

CONFIG = {
    "start_order_list": ['steak'] * 4,
    "cook_time": 10,
    "delivery_reward": 20,
    'num_items_for_steak': 1,
    'chop_time': 2,
    'wash_time': 2,
    "rew_shaping_params": None
}

def load_steak_qmdp_agent(env, agent_save_path, value_kb_save_path, lvl_config):
    ai_agent = None
    print(agent_save_path, value_kb_save_path)
    if agent_save_path is not None:
        # agent saved before, load it.
        if os.path.exists(agent_save_path):
            with open(agent_save_path, 'rb') as f:
                ai_agent = pickle.load(f)
        if value_kb_save_path is not None:
            # agent saved before, load it.
            if os.path.exists(value_kb_save_path):
                with open(value_kb_save_path, 'rb') as f:
                    [ai_agent.mdp_planner.world_state_cost_dict, ai_agent.mdp_planner.track_state_kb_map] = pickle.load(f)
    
    # agent not found, recreate it and save it if a path is given.
    if ai_agent == None:
        ai_agent = init_steak_qmdp_agent(env, search_depth=lvl_config['search_depth'], kb_search_depth=lvl_config['kb_search_depth'], vision_limit=lvl_config['vision_limit'], vision_bound=lvl_config['vision_bound'], kb_update_delay=lvl_config['kb_update_delay'], vision_limit_aware=lvl_config['vision_limit_aware'])
        ai_agent.set_agent_index(0)
        if agent_save_path is not None:
            with open(agent_save_path, 'wb') as f:
                pickle.dump(ai_agent, f)
    return ai_agent

def load_human_log_data(log_index):
    human_log_csv = os.path.join(LSI_STEAK_STUDY_RESULT_DIR, log_index,
                                 "human_log.csv")
    if not os.path.exists(human_log_csv):
        print("Log dir does not exit.")
        exit(1)
    human_log_data = pd.read_csv(human_log_csv)
    return human_log_csv, human_log_data

def load_analyzed_data(log_index):
    human_log_csv = os.path.join(LSI_STEAK_STUDY_RESULT_DIR, log_index,
                                 "analysis.csv")
    if not os.path.exists(human_log_csv):
        print("Log dir does not exit.")
        exit(1)
    human_log_data = pd.read_csv(human_log_csv)
    return human_log_csv, human_log_data

def kb_diff_measure(human_kb, world_kb):
    diff_count, diff_total = 0, 0
    kb_diff = []
    diff_measure = np.zeros((len(human_kb),4), dtype=int)
    for i, (h, w) in enumerate(zip(human_kb, world_kb)):
        h_obj = h.split('.')
        w_obj = w.split('.')

        for j, (ho, wo) in enumerate(zip(h_obj, w_obj)):
            if j < 3:
                diff = abs(int(ho) - int(wo))
                if diff == 3: diff = 1
                diff_measure[i][j] = diff
            else:
                if ho != wo:
                    diff_measure[i][j] = 1
        
        kb_diff.append(sum(diff_measure[i]))
        if sum(diff_measure[i]) > 0: 
            diff_count += 1
            diff_total += sum(diff_measure[i])
            

    # print(diff_measure)
    return diff_measure, kb_diff, round(diff_count/len(human_kb)*100, 2), round(diff_total/diff_count, 2)

def obj_held_freq(robot_holding_log, human_holding_log):
    obj_held_freq_dict = {}
    obj_diff_dict = {}
    obj_list = ['onion', 'meat', 'plate', 'hot_plate', 'steak', 'dish']
    for obj_name in obj_list:
        obj_held_freq_dict[obj_name] = [robot_holding_log[obj_name], human_holding_log[obj_name]]
        obj_diff_dict[obj_name] = (robot_holding_log[obj_name] - human_holding_log[obj_name])
    prep_count = [robot_holding_log['onion'] + robot_holding_log['meat'] + robot_holding_log['plate'], human_holding_log['onion'] + human_holding_log['meat'] + human_holding_log['plate']]
    prep_count_diff = prep_count[0] - prep_count[1]
    plating_count = [robot_holding_log['hot_plate'] + robot_holding_log['steak'] + robot_holding_log['dish'],human_holding_log['hot_plate'] + human_holding_log['steak'] + human_holding_log['dish']]
    plating_count_diff = plating_count[0] - plating_count[1]

    return obj_held_freq_dict, obj_diff_dict, prep_count, prep_count_diff, plating_count, plating_count_diff

def subtask_analysis(subtasks):
    stop_count = subtasks.count(0)
    turn_count = subtasks.count(1)
    stay_count = subtasks.count(2)
    interact_count = subtasks.count(3)
    interupt_freq = (stop_count+turn_count+stay_count+interact_count)/len(subtasks)

    interrupt_occurance = 0
    prev_s = 'None'
    change_of_subtasks = 0
    undo_counts = 0
    check_counts = 0
    for s in subtasks:
        if prev_s in [0,1] and s == 3: # does an interact to place things down
            undo_counts += 1
        if prev_s in [0,1] and s not in [0,1,2,3]: # does a turn or stop and go back to subtask
            check_counts += 1
        if s in [0,1] and prev_s not in [0,1,2,3]:
            interrupt_occurance += 1
        if s != prev_s and s not in [0,1,2,3]:
            change_of_subtasks += 1
        prev_s = s
    total_different_subtasks = interrupt_occurance + change_of_subtasks
    return stop_count, turn_count, stay_count, interact_count, undo_counts, check_counts, round(interupt_freq*100, 2), round((interrupt_occurance/total_different_subtasks)*100, 2), round(undo_counts/total_different_subtasks*100,2), round(check_counts/total_different_subtasks*100,2)

def plot_kb_and_subtasks(joint_action, subtask_log, human_kb_log, world_kb_log, robot_holding_log, human_holding_log, robot_in_bound_count, log_dir=None, log_name=None, log_index=None):
    if len(subtask_log) == 0:
        return

    img_dir = os.path.join(log_dir, log_name)

    # Generate some example data
    time = np.arange(0, len(joint_action))  # Time values from 0 to 200
    task_labels = ['stop', 'up/down/left/right', 'stay', 'interact', 'pickup meat', 'drop meat', 'pickup onion', 'drop onion', 'chop onion', 'pickup plate', 'drop plate', 'heat hot plate', 'pickup hot plate', 'pickup steak', 'pickup garnish', 'deliver dish']
    bar_values = np.random.rand(201) * 5  # Random values for the bottom row (0-5)

    subtask_values = []
    for s in subtask_log:
        if s not in ['up', 'down', 'right', 'left']:
            subtask_values.append(task_labels.index(s.replace('_',' ')))
        else:
            subtask_values.append(1)
    # Create a grid of subplots with shared x-axis and no space in between rows
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3.5, 1.25, 1])  # 3 rows, 1 column
    # Adjust the height ratios to make the top graph 1/3 of the height of the bottom graphs

    # Plot the top row (dot graph)
    ax0 = plt.subplot(gs[0])
    ax0.scatter(time, subtask_values, marker='.', color='brown')
    ax0.set_yticks(range(len(task_labels)))
    ax0.set_yticklabels(task_labels)
    ax0.set_ylabel('Human selected tasks')
    ax0.set_title('Actions and Knowledge Difference Log')
    ax0.set_ylim(-1, len(task_labels))
    ax0.grid(True, linestyle=':', alpha=0.3)

    actions = [(0,-1), (0,1), (-1, 0), (1,0), (0,0), 'interact']
    action_labels = ['Up', 'Down', 'Left', 'Right', 'Stay', 'Interact']
    action_log = []
    j = np.array(joint_action, dtype=object)
    for a in j[:,0]:
        action_log.append(actions.index(a))
        
    ax2 = plt.subplot(gs[1], sharex=ax0)
    ax2.scatter(time, action_log, marker='s', s=9, color='purple')
    ax2.set_yticks(range(len(action_labels)))
    ax2.set_yticklabels(action_labels)
    ax2.set_ylabel('Robot actions')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.set_ylim(-1, len(action_labels))
    ax2.grid(True, linestyle=':', alpha=0.3)

    kb_diff, kb_diff_list, kb_diff_freq, kb_diff_avg = kb_diff_measure(human_kb_log, world_kb_log)

    # Plot the bottom row (bar graph)
    ax1 = plt.subplot(gs[2], sharex=ax0)  # Share x-axis with the top row
    if len(kb_diff) > 0:
        b1 = ax1.bar(time, kb_diff[:,0], label='num_item_in_grill', align='center')
        b2 = ax1.bar(time, kb_diff[:,1], bottom=kb_diff[:,0], label='garnish status', align='center')
        b3 = ax1.bar(time, kb_diff[:,2], bottom=kb_diff[:,0]+kb_diff[:,1], label='plate status', align='center')
        b4 = ax1.bar(time, kb_diff[:,3], bottom=kb_diff[:,0]+kb_diff[:,1]+kb_diff[:,2], label='robot held object', align='center')

        handles = [b1, b2, b3, b4]
        labels = ['num_item_in_grill', 'garnish status', 'plate status', 'robot held object']
        fig.legend(handles, labels,loc='upper left', bbox_to_anchor=(0.141, 0.88))

    # ax1.bar(time, kb_diff, color='red', width=1.0)
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('KB. diff.')
    ax1.set_ylim(0, 5)
    # ax1.set_title('Bottom Row: Bar Graph')
    # ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.3)


    # Remove space between subplots
    plt.subplots_adjust(hspace=0)
    # plt.tight_layout()
    plt.savefig(img_dir+'_kb_subtask.png')

    # Display the plot
    # plt.show()

    stop_count, turn_count, stay_count, interact_count, undo_count, check_count, interrupt_freq, interrupt_occurance, undo_freq, check_freq = subtask_analysis(subtask_values)
    obj_held_freq_dict, obj_diff_dict, prep_count, prep_count_diff, plating_count, plating_count_diff = obj_held_freq(robot_holding_log, human_holding_log)
    robot_inbound_freq = round(sum(robot_in_bound_count)/len(robot_in_bound_count)*100, 2)

    results = [len(subtask_values), interrupt_freq, interrupt_occurance, undo_freq, check_freq, kb_diff_freq, kb_diff_avg, robot_inbound_freq, prep_count, prep_count_diff, plating_count, plating_count_diff, kb_diff_list, robot_in_bound_count, stop_count, turn_count, stay_count, interact_count, undo_count, check_count, obj_held_freq_dict, obj_diff_dict]

    return results

def replay_with_joint_actions(lvl_str, joint_actions, plot=True, log_dir=None, log_name=None, view_angle=0, agent_save_path = None, human_save_path=None):
    """Replay a game play with given level and joint actions.

    Args:
        joint_actions (list of tuple of tuple): Joint actions.
    """
    grid = lvl_str2grid(lvl_str)
    mdp = SteakHouseGridworld.from_grid(grid, CONFIG)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=HUMAN_STUDY_ENV_HORIZON)
    tmp_ai_agent=None
    if agent_save_path is not None:
        # agent saved before, load it.
        if os.path.exists(agent_save_path):
            with open(agent_save_path, 'rb') as f:
                tmp_ai_agent = pickle.load(f)

    tmp_human_agent=None
    if human_save_path is not None:
        # agent saved before, load it.
        if os.path.exists(human_save_path):
            with open(human_save_path, 'rb') as f:
                tmp_human_agent = pickle.load(f)
                tmp_human_agent.set_agent_index(1)
    done = False
    t = 0

    img_dir = os.path.join(log_dir, log_name)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_name = lambda timestep: f"{img_dir}/{t:05d}.png"

    # Hacky: use human agent for replay.
    ai_agent = HumanPlayer()
    player = HumanPlayer()

    ai_agent.set_agent_index(0)
    ai_agent.set_mdp(env.mdp)
    player.set_agent_index(1)
    player.set_mdp(env.mdp)
    i = 0
    last_state = None
    total_sparse_reward = 0
    checkpoints = [env.horizon - 1] * env.num_orders
    cur_order = 0
    world_kb_log = []
    robot_holding = {'meat': 0, 'onion': 0, 'plate': 0, 'hot_plate': 0, 'steak': 0, 'dish':0}
    human_holding = {'meat': 0, 'onion': 0, 'plate': 0, 'hot_plate': 0, 'steak': 0, 'dish':0}
    prev_robot_hold = 'None'
    prev_human_hold = 'None'
    robot_in_bound_count = []
    subtask_log=[]

    while i < len(joint_actions):

        if plot:
            if view_angle > 0: 
                env.render("fog", view_angle=view_angle)
            else:
                env.render()
            time.sleep(0.2)
            
            if img_name is not None:
                cur_name = img_name(t)
                pygame.image.save(env.mdp.viewer, cur_name)

        if agent_save_path is not None: world_kb_log += tmp_ai_agent.mdp_planner.update_world_kb_log(env.state)

        if human_save_path is not None: 
            robot_in_bound_count.append(tmp_human_agent.in_bound(env.state, env.state.players[0].position, vision_bound=120/2))
            tmp_human_agent.update(env.state)
            tmp_human_agent.update_kb_log()

        if env.state.players[0].held_object is not None:
            obj_name = env.state.players[0].held_object.name
            if obj_name != prev_robot_hold:
                if obj_name not in robot_holding.keys():
                    robot_holding[obj_name] = 1
                else:
                    robot_holding[obj_name] += 1
            prev_robot_hold = obj_name
        else:
            prev_robot_hold = 'None'
        
        if env.state.players[1].held_object is not None:
            obj_name = env.state.players[1].held_object.name
            if obj_name != prev_human_hold:
                if obj_name not in human_holding.keys():
                    human_holding[obj_name] = 1
                else:
                    human_holding[obj_name] += 1
            prev_human_hold = obj_name
        else:
            prev_human_hold = 'None'
                
        total_sparse_reward += timestep_sparse_reward

        if timestep_sparse_reward > 0:
            checkpoints[cur_order] = i
            cur_order += 1
        
        i += 1
        t += 1

    if plot: os.system("ffmpeg -y -r 5 -i \"{}%*.png\"  {}{}.mp4".format(img_dir+'/', log_dir+'/', log_name))
    # os.system("ffmpeg -r 5 -i \"{}%*.png\"  {}{}.mp4".format(img_dir+'/', log_dir+'/', log_name))
    shutil.rmtree(img_dir) 

    return world_kb_log, human_kb_log, subtask_log, robot_holding, human_holding, robot_in_bound_count

def load_steak_human_agent(env, human_save_path, vision_bound):
    human_agent = None
    if human_save_path is not None:
        # agent saved before, load it.
        if os.path.exists(human_save_path):
            with open(human_save_path, 'rb') as f:
                human_agent = pickle.load(f)
        else:
            mlp = planners.MediumLevelPlanner.from_pickle_or_compute(env.mdp, BASE_PARAMS, force_compute=True)

            human_agent = agent.SteakLimitVisionHumanModel(mlp, env.state, auto_unstuck=True, explore=False, vision_limit=True, vision_bound=vision_bound, kb_update_delay=1, debug=False)
            human_agent.set_agent_index(1)

            if human_save_path is not None:
                with open(human_save_path, 'wb') as f:
                    pickle.dump(human_agent, f)

    return human_agent

def read_in_study_lvl(_dir):
    """Read in levels used for human study in the given directory."""
    lvls = []
    for i, _file in enumerate(os.listdir(_dir)):
        if _file.endswith(".tml"):
            lvl = toml.load(os.path.join(_dir, _file))
            lvls.append(lvl)
    return lvls

def write_row(csv_file, to_add):
    """Append a row to csv file"""
    with open(csv_file, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(to_add)

def write_to_human_exp_log(lvl_type_full, results, lvl_config):
    """Write to human exp log.

    Args:
        human_log_csv (str): Path to the human_log.csv file.
        results (tuple) all of the results returned from the human study.
        lvl_config (dic): toml config dic of the level.
    """
    assert os.path.exists(human_log_csv)

    to_write = [
        lvl_config["lvl_type"] if "lvl_type" in lvl_config else None,
        lvl_config["ID"] if "ID" in lvl_config else None,
        lvl_config["vision_limit"] if "vision_limit" in lvl_config else None,
        lvl_config["vision_bound"] if "vision_bound" in lvl_config else None,
        lvl_config["vision_limit_aware"] if "vision_limit_aware" in lvl_config else None,
        lvl_config["search_depth"] if "search_depth" in lvl_config else None,
        lvl_config["kb_search_depth"] if "kb_search_depth" in lvl_config else None,
        lvl_config["kb_update_delay"] if "kb_update_delay" in lvl_config else None,
        *results,
        lvl_config["lvl_str"],
    ]

    write_row(human_log_csv, to_write)

def create_human_exp_log():
    """ Create human_study/result/<exp_id>. <exp_id> would be determined by the
        first digit that does not exist under the human_exp directory.

        Returns:
            Path to the csv file to which the result is be written.
    """
    # increment from 0 to find a directory name that does not exist yet.
    exp_dir = 0
    while os.path.isdir(os.path.join(LSI_STEAK_STUDY_RESULT_DIR,
                                     str(exp_dir))):
        exp_dir += 1
    exp_dir = os.path.join(LSI_STEAK_STUDY_RESULT_DIR, str(exp_dir))
    os.mkdir(exp_dir)

    # create csv file to store results
    human_log_csv = os.path.join(exp_dir, 'human_log.csv')

    # construct labels
    data_labels = [
        "lvl_type",
        "ID",
        "vision_limit",
        "vision_bound",
        "vision_limit_aware",
        "search_depth",
        "kb_search_depth",
        "kb_update_delay",
        "complete",
        "joint_actions",
        "total time steps",
        "subtask_log",
        "human_kb_log",
        "world_kb_log",
        "num_subtask_actions",
        "lvl_str",
    ]

    write_row(human_log_csv, data_labels)

    return human_log_csv

def write_to_analysis_log(path, results, lvl_name, id):
    """Write to human exp log.

    Args:
        human_log_csv (str): Path to the human_log.csv file.
        results (tuple) all of the results returned from the human study.
        lvl_config (dic): toml config dic of the level.
    """
    assert os.path.exists(path)

    to_write = [
        lvl_name,
        id,
        *results,
    ]

    write_row(path, to_write)

def create_analysis_log(log_id):
    """ Create human_study/result/<exp_id>. <exp_id> would be determined by the
        first digit that does not exist under the human_exp directory.

        Returns:
            Path to the csv file to which the result is be written.
    """
    human_log_csv = os.path.join(LSI_STEAK_STUDY_RESULT_DIR, str(log_id)+'/analysis_log.csv')

    if os.path.exists(human_log_csv):
        os.remove(human_log_csv)

    # construct labels
    data_labels = [
        "lvl_type",
        "ID",
        "total_steps",
        "interrupt_freq",
        "interrupt_occurance",
        "undo_freq",
        "check_freq",
        "kb_diff_freq",
        "kb_diff_avg",
        'robot_inbound_freq',
        "prep_count",
        "prep_count_diff",
        "plating_count",
        "plating_count_diff",
        "kb_diff_list",
        'robot_in_bound_count',
        "stop_count",
        "turn_count",
        "stay_count",
        "interact_count",
        "undo_count",
        "check_count",
        "obj_held_freq_dict",
        "obj_diff_dict",
    ]

    write_row(human_log_csv, data_labels)

    return human_log_csv


def gen_save_pths(lvl_config):
    agent_save_path = os.path.join(
        LSI_STEAK_STUDY_AGENT_DIR, "{lvl_type}_{lvl_vision}_{lvl_robot_aware}_{lvl_search_depth}_{lvl_kb_depth}_{kb_update_delay}.pkl".format(lvl_type=lvl_config["lvl_type"], lvl_vision=lvl_config["vision_bound"], lvl_robot_aware=lvl_config["vision_limit_aware"], lvl_search_depth=lvl_config["search_depth"], lvl_kb_depth=lvl_config["kb_search_depth"], kb_update_delay=lvl_config['kb_update_delay']))
    value_kb_save_path = os.path.join(
        LSI_STEAK_STUDY_AGENT_DIR,
        "{lvl_type}_{lvl_vision}_{lvl_robot_aware}_{lvl_search_depth}_{lvl_kb_depth}_{kb_update_delay}_v_and_kb.pkl".format(lvl_type=lvl_config["lvl_type"], lvl_vision=lvl_config["vision_bound"], lvl_robot_aware=lvl_config["vision_limit_aware"], lvl_search_depth=lvl_config["search_depth"], lvl_kb_depth=lvl_config["kb_search_depth"], kb_update_delay=lvl_config['kb_update_delay']))
    human_save_path = os.path.join(
        LSI_STEAK_STUDY_AGENT_DIR,
        "{lvl_type}_{lvl_vision}_{lvl_robot_aware}_{lvl_search_depth}_{lvl_kb_depth}_{kb_update_delay}_human.pkl".format(lvl_type=lvl_config["lvl_type"], lvl_vision=lvl_config["vision_bound"], lvl_robot_aware=lvl_config["vision_limit_aware"], lvl_search_depth=lvl_config["search_depth"], lvl_kb_depth=lvl_config["kb_search_depth"], kb_update_delay=lvl_config['kb_update_delay']))
    return agent_save_path, value_kb_save_path, human_save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--log_index',
                        help='Integer: index of the study log',
                        required=False,
                        default=-1)
    parser.add_argument('--gen_vid',
                        action='store_true',
                        help='Whether to continue running a previous study',
                        default=False)
    parser.add_argument('--gen_plot',
                        action='store_true',
                        help='Whether to continue running a previous study',
                        default=False)
    opt = parser.parse_args()


    log_index = opt.log_index
    _, human_log_data = load_human_log_data(log_index)
    NO_FOG_COPY=True

    # play all of the levels
    analysis_log_csv = create_analysis_log(opt.log_index)

    for index, lvl_config in human_log_data.iterrows():
        # get level string and logged joint actions from log file
        lvl_str = lvl_config["lvl_str"]
        joint_actions = ast.literal_eval(lvl_config["joint_actions"])

        # replay the game
        if opt.gen_vid:
            agent_save_path, _, human_save_path = gen_save_pths(lvl_config)
            tmp_world_kb_log, tmp_human_kb_log, tmp_subtask_log, robot_holding_log, human_holding_log, robot_in_bound_count = replay_with_joint_actions(lvl_str, joint_actions, log_dir=os.path.join(LSI_STEAK_STUDY_RESULT_DIR, log_index), log_name=lvl_config["lvl_type"]+'_'+str(lvl_config["kb_search_depth"]), view_angle=lvl_config["vision_bound"], agent_save_path=agent_save_path, human_save_path=human_save_path)

            results = plot_kb_and_subtasks(joint_actions, tmp_subtask_log, tmp_human_kb_log, world_kb_log, robot_holding_log, human_holding_log, robot_in_bound_count, log_dir=os.path.join(LSI_STEAK_STUDY_RESULT_DIR, log_index), log_name=lvl_config["lvl_type"]+'_'+str(lvl_config["kb_search_depth"]), log_index=log_index)

            write_to_analysis_log(analysis_log_csv, results, lvl_config["lvl_type"]+'_'+str(lvl_config["kb_search_depth"]), log_index)               

            if NO_FOG_COPY and lvl_config["vision_bound"] > 0:
                replay_with_joint_actions(lvl_str, joint_actions, log_dir=os.path.join(LSI_STEAK_STUDY_RESULT_DIR, log_index), log_name=lvl_config["lvl_type"]+'_'+str(lvl_config["kb_search_depth"])+'_nofog', view_angle=0)

