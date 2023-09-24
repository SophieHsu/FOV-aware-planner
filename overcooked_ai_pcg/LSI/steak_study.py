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

SUB_STUDY_TYPES = [
    'even_workloads',
    'uneven_workloads',
    'high_team_fluency',
    'low_team_fluency',
]

SUB_STUDY_TYPES_VALUE_RANGE = [
    [i for i in range(-6,7,1)],
    [i for i in range(-6,7,1)],
    [i for i in range(0,101,1)],
    [i for i in range(0,101,1)]
]

NON_TRIAL_STUDY_TYPES = [
    'all',
    *SUB_STUDY_TYPES,
]

SAME_AGENT_STUDY_TYPES = [
    'Side-3-120',#use
    'Side-3_120_not_aware',
    'Mid-2-120', #smooth
    'Mid-2_120_not_aware', #smooth (acan)
    'Mid-1_120_not_aware', #tied (wasn't sure about the sink pick up)
    'Mid-1-120', #more sure, so can use. #no 
    'Side-1-120', # super confussed in the beginning # nope
    'Side-1_120_not_aware', # i prefer this one
    'Mid-3_120_not_aware', #looked twice to check agent
    'Mid-3-120',
    'None-3_120_not_aware', #like
    'None-3-120', # like more as the agent'
]

DETAILED_STUDY_TYPES = [ 
    'None-1-120',
    'None-1_120_not_aware',
    'Side-3-120',#use
    'Side-3_120_not_aware',
    'Mid-2-120', #smooth
    'Mid-2_120_not_aware', #smooth (acan)
    'Side-2_120_not_aware', #cant tell #use as test
    'Side-2-120',
    'None-2-120', # gets stuck at grill 
    'None-2_120_not_aware', #tied # i prefer this
    'Mid-1_120_not_aware', #tied (wasn't sure about the sink pick up)
    'Mid-1-120', #more sure, so can use. #no 
    'Side-1-120', # super confussed in the beginning # nope
    'Side-1_120_not_aware', # i prefer this one
    'Mid-3_120_not_aware', #looked twice to check agent
    'Mid-3-120',
    'None-3_120_not_aware', #like
    'None-3-120', # like more as the agent does show up in sight
]

USER_STUDY_AWARE_TYPES = [ 
    'Side-3-120',#use
    'Mid-2-120', #smooth
    'Mid-1-120', #more sure, so can use. #no 
    'Mid-3-120',
    'None-3-120', # like more as the agent does show up in sight
]

RELOAD_STUDY_TYPE0 = [
    'Mid-2-120',
    'Side-3-120',
    'Mid-1-120',
    'None-3-120',
    'Mid-1_120_not_aware',
    'None-3_120_not_aware',
    'Side-3_120_not_aware',
    'Mid-2_120_not_aware',
]

RELOAD_STUDY_TYPE1 = [
    'Mid-1_120_not_aware',
    'None-3_120_not_aware',
    'Side-3_120_not_aware',
    'Mid-2_120_not_aware',
    'Mid-2-120',
    'Side-3-120',
    'Mid-1-120',
    'None-3-120',
]

USER_STUDY_UNAWARE_TYPES = [ 
    'Side-3_120_not_aware',
    'Mid-2_120_not_aware', #smooth (acan)
    'Mid-1_120_not_aware', #tied (wasn't sure about the sink pick up)
    'Mid-3_120_not_aware', #looked twice to check agent
    'None-3_120_not_aware', #like
]

USER_STUDY_AWARE_TYPES = [ 
    'Side-3-120',#use
    'Mid-2-120', #smooth
    'Mid-1-120', #more sure, so can use. #no 
    'Mid-3-120',
    'None-3-120', # like more as the agent does show up in sight
]

USER_STUDY_UNAWARE_TYPES = [ 
    'Side-3_120_not_aware',
    'Mid-2_120_not_aware', #smooth (acan)
    'Mid-1_120_not_aware', #tied (wasn't sure about the sink pick up)
    'Mid-3_120_not_aware', #looked twice to check agent
    'None-3_120_not_aware', #like
]

ALL_STUDY_TYPES = [
    'all',
    'trial',
    *SUB_STUDY_TYPES,
]

CONFIG = {
    "start_order_list": ['steak'] * 2,
    "cook_time": 10,
    "delivery_reward": 20,
    'num_items_for_steak': 1,
    'chop_time': 2,
    'wash_time': 2,
    "rew_shaping_params": None
}

class OvercookedGame:
    """Class for human player to play an Overcooked game with given AI agent.
       Most of the code from http://pygametutorials.wikidot.com/tutorials-basic.

    Args:
        env: Overcooked Environment.
        agent: AI agent that the human plays game with.
        rand_seed: random seed.
        agent_idx: index of the AI agent.
        slow_time (bool): whether the AI agent take multiple actions each time
            the human take one action.
    """
    def __init__(self, env, agent, agent_idx, rand_seed, human_agent=None, slow_time=False, fog=False, trial=False, vision_aware=True):
        self._running = True
        self._display_surf = None
        self.env = env
        self.trial = trial
        self.agent = agent
        self.agent_idx = agent_idx
        self.human_player = HumanPlayer()
        self.slow_time = slow_time
        self.total_sparse_reward = 0
        self.last_state = None
        self.rand_seed = rand_seed
        self.fog = fog
        self.path = []
        self.next_path = []
        self.curr_s = 'None'
        self.next_s = 'None'
        self.human_agent = human_agent
        self.vision_aware = vision_aware
        self.loop_time_start = time.time()
        self.subtask_log = []
        self.selected_action_count=0

        # Saves when each soup (order) was delivered
        self.checkpoints = [env.horizon - 1] * env.num_orders
        self.cur_order = 0
        self.timestep = 0
        self.joint_actions = []
        self.step_time = 1

    def on_init(self):
        pygame.init()

        # Adding AI agent as teammate
        self.agent.set_agent_index(self.agent_idx)
        self.agent.set_mdp(self.env.mdp)
        self.human_player.set_agent_index(1 - self.agent_idx)
        self.human_player.set_mdp(self.env.mdp)

        self.env.render(mode="right_panel_init", view_angle=self.human_agent.vision_bound)
        for box in self.env.mdp.rend_boxes:
            box.render_checkbox()
        self._running = True
        np.random.seed(self.rand_seed)

    def on_event(self, event):
        done = False
        next_state = None

        if event.type == pygame.KEYDOWN:
            pressed_key = event.dict['key']
            action = None

            if pressed_key == pygame.K_UP:
                action = Direction.NORTH
            elif pressed_key == pygame.K_RIGHT:
                action = Direction.EAST
            elif pressed_key == pygame.K_DOWN:
                action = Direction.SOUTH
            elif pressed_key == pygame.K_LEFT:
                action = Direction.WEST
            elif pressed_key == pygame.K_SPACE:
                action = Action.INTERACT
            elif pressed_key == pygame.K_s:
                action = Action.STAY

            if action in Action.ALL_ACTIONS:

                done, next_state = self.step_env(action)

                if self.slow_time and not done:
                    for _ in range(2):
                        action = Action.STAY
                        done, next_state = self.step_env(action)
                        if done:
                            break

        if event.type == pygame.MOUSEBUTTONDOWN:    
            for box in self.env.mdp.rend_boxes:
                box.update_checkbox(event)
                if box.checked is True:
                    # only update when the task change
                    if (self.next_s != box.caption) or (self.next_s in ['chop onion', 'heat hot plate', 'up', 'down', 'left', 'right', 'stay', 'interact']) :
                        self.selected_action_count += 1
                        self.next_s = box.caption
                        if self.next_s in ['up', 'down', 'left', 'right', 'stay', 'interact', 'stop']:
                            if 'up' in self.next_s:
                                self.next_path = [Direction.NORTH]
                            elif 'down' in self.next_s:
                                self.next_path = [Direction.SOUTH]
                            elif 'left' in self.next_s:
                                self.next_path = [Direction.WEST]
                            elif 'right' in self.next_s:
                                self.next_path = [Direction.EAST]
                            elif 'stay' in self.next_s:
                                self.next_path = [Action.STAY]
                            elif 'stop' in self.next_s:
                                self.curr_s = self.next_s
                                self.path = [Action.STAY]
                            elif 'interact' in self.next_s:
                                self.next_path = [Action.INTERACT]
                        else:
                            s = '_'.join(self.next_s.split())
                            _, _, self.next_path = self.human_agent.action(self.env.state, chosen_subtask=s, return_path=True)
                            
                    for b in self.env.mdp.rend_boxes:
                        if b != box:
                            b.checked = False
        
        for box in self.env.mdp.rend_boxes:
            box.render_checkbox()

        if event.type == pygame.QUIT or done:
            self._running = False
            self.last_state = next_state

    def step_env(self, my_action, curr_s):
        agent_action = self.agent.action(self.env.state)[0]

        if self.agent_idx == 1:
            joint_action = (my_action, agent_action)
        else:
            joint_action = (agent_action, my_action)

        self.joint_actions.append(joint_action)
        next_state, timestep_sparse_reward, done, info = self.env.step(
            joint_action)

        # update logs of human player for bc calculations
        self.human_player.update_logs(next_state, my_action, curr_s)
        self.human_agent.update(self.env.state)
        self.human_agent.update_kb_log()

        if timestep_sparse_reward > 0:
            self.checkpoints[self.cur_order] = self.timestep
            self.cur_order += 1

        self.total_sparse_reward += timestep_sparse_reward
        self.timestep += 1
        # print("Timestep:", self.timestep)
        return done, next_state

    def on_loop(self):
        replanned = False

        # recheck if the path is done before interact
        if len(self.path) == 1 and self.path[0] == 'interact' and self.curr_s not in ['interact']:
            s = '_'.join(self.curr_s.split())
            _, _, self.path = self.human_agent.action(self.env.state, chosen_subtask=s, return_path=True)
            if len(self.path) > 1:
                self.path = [Action.STAY] + self.path
                replanned = True
        
        # if len(self.path) == 0 and self.curr_s not in ['chop onion', 'heat hot plate', 'turn up', 'turn down', 'turn left', 'turn right', 'stay', 'interact', 'None']:

        if len(self.path) == 0:
            s = '_'.join(self.next_s.split())
            if self.next_s not in ['up', 'down', 'left', 'right', 'stay', 'interact', 'None']:
                _, _, self.next_path = self.human_agent.action(self.env.state, chosen_subtask=s, return_path=True)

            self.path = self.next_path
            self.next_path = []
            self.curr_s = self.next_s
            self.next_s = 'None'

            if len(self.next_path) == 0:
                for box in self.env.mdp.rend_boxes:
                    box.checked = False
                    box.render_checkbox()
        
        if len(self.path) > 0:
            human_action = self.path.pop(0)
            done, next_state = self.step_env(human_action, self.curr_s)

            if done:
                self._running = False
                self.last_state = next_state
                if not self.trial: self.subtask_log = self.env.state.players[self.human_agent.agent_index].subtask_log

    def on_render(self):
        # if self.fog:
        # time.sleep(max(0, self.step_time - time.time()-self.loop_time_start))
        if not self.trial and self.vision_aware:
            self.env.render(mode="right_panel", view_angle=self.human_agent.vision_bound, info=self.curr_s, selected_action_count=self.selected_action_count)
        else:
            self.env.render(mode="not_aware", view_angle=self.human_agent.vision_bound, info=self.curr_s, selected_action_count=self.selected_action_count)
        time.sleep(0.2)
        # self.loop_time_start = time.time()
        # else:
        #     self.env.render()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while (self._running):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_render()
            self.on_loop()
        self.on_cleanup()
        # workloads = self.last_state.get_player_workload()
        # concurr_active = self.last_state.cal_concurrent_active_sum()
        # stuck_time = self.last_state.cal_total_stuck_time()

        # from IPython import embed
        # embed()

        fitness = self.total_sparse_reward + 1
        for checked_time in reversed(self.checkpoints):
            fitness *= self.env.horizon
            fitness -= checked_time

        if not self.trial:
            # if value_kb_save_path is not None:
            #     with open(value_kb_save_path, 'wb') as f:
            #         pickle.dump([self.agent.mdp_planner.world_state_cost_dict, self.agent.mdp_planner.track_state_kb_map], f)

            return (self.total_sparse_reward, self.joint_actions, len(self.joint_actions), self.subtask_log, self.human_agent.kb_log, self.agent.mdp_planner.world_kb_log, self.selected_action_count)
        else:
            return (self.total_sparse_reward, self.joint_actions, len(self.joint_actions))

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

def kb_diff_measure(human_kb, world_kb):
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

    # print(diff_measure)
    return diff_measure

def plot_kb_and_subtasks(joint_action, subtask_log, human_kb_log, world_kb_log, log_dir=None, log_name=None):
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
            subtask_values.append(task_labels.index(s))
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

    kb_diff = kb_diff_measure(human_kb_log, world_kb_log)

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


def replay_with_joint_actions(lvl_str, joint_actions, plot=True, log_dir=None, log_name=None, view_angle=0):
    """Replay a game play with given level and joint actions.

    Args:
        joint_actions (list of tuple of tuple): Joint actions.
    """
    grid = lvl_str2grid(lvl_str)
    mdp = SteakHouseGridworld.from_grid(grid, CONFIG)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=HUMAN_STUDY_ENV_HORIZON)
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

    while not done:

        if plot:
            if view_angle > 0: 
                env.render("fog", view_angle=view_angle)
            else:
                env.render()
            time.sleep(0.2)
            
            if img_name is not None:
                cur_name = img_name(t)
                pygame.image.save(env.mdp.viewer, cur_name)

        # if t == 0:
        #     pygame.image.save(
        #     env.mdp.viewer,
        #     os.path.join(log_dir, log_name+".png"))

        ai_agent.update_logs(env.state, joint_actions[i][0])
        player.update_logs(env.state, joint_actions[i][1])
        next_state, timestep_sparse_reward, done, info = env.step(
            joint_actions[i])
        total_sparse_reward += timestep_sparse_reward

        if timestep_sparse_reward > 0:
            checkpoints[cur_order] = i
            cur_order += 1
        # print(joint_actions[i])
        last_state = next_state
        i += 1
        t += 1

    os.system("ffmpeg -r 5 -i \"{}%*.png\"  {}{}.mp4".format(img_dir+'/', log_dir+'/', log_name))
    shutil.rmtree(img_dir) 

    # pygame.image.save(
    #     env.mdp.viewer,
    #     os.path.join(log_dir, log_name+".png"))

    # recalculate the bcs
    workloads = next_state.get_player_workload()
    concurr_active = next_state.cal_concurrent_active_sum()
    stuck_time = next_state.cal_total_stuck_time()
    return workloads, concurr_active, stuck_time, checkpoints, i


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

def human_play(
    lvl_str,
    ai_agent=None,
    human_save_path=None,
    agent_save_path=None,
    value_kb_save_path=None,
    horizon=HUMAN_STUDY_ENV_HORIZON,
    trial = False,
    lvl_config=None
):
    """Function that allows human to play with an ai_agent.

    Args:
        lvl_str (str): Level string.
        ai_agent (Agent): Agent that human plays with. Default is QMDP agent.
        agent_save_path (str): Path to the pre-saved ai agent. If nothing is
            found, it will be saved to there.
        horizon (int): max number of timesteps to play.
    """
    env = init_steak_env(lvl_str, horizon=horizon)
    if ai_agent is None:
        ai_agent = load_steak_qmdp_agent(env, agent_save_path, value_kb_save_path, lvl_config)
    if trial:
        human_agent = load_steak_human_agent(env, human_save_path, vision_bound=lvl_config['vision_bound'])
    else:
        human_agent = load_steak_human_agent(env, human_save_path, vision_bound=120)
    theApp = OvercookedGame(env, ai_agent, agent_idx=0, rand_seed=10, human_agent=human_agent, trial=trial, vision_aware=lvl_config['vision_limit_aware'])
    return theApp.on_execute()


def agents_play(
    lvl_str,
    ai_agent=None,
    agent_save_path=None,
    value_kb_save_path = None,
    horizon=HUMAN_STUDY_ENV_HORIZON,
    VISION_LIMIT = True,
    VISION_BOUND = 120,
    VISION_LIMIT_AWARE = True,
    EXPLORE = False,
    agent_unstuck = True,
    human_unstuck = True,
    SEARCH_DEPTH = 5,
    KB_SEARCH_DEPTH = 3,
    KB_UPDATE_DELAY=3,
):
    """Function that allows human to play with an ai_agent.

    Args:
        lvl_str (str): Level string.
        ai_agent (Agent): Agent that human plays with. Default is QMDP agent.
        agent_save_path (str): Path to the pre-saved ai agent. If nothing is
            found, it will be saved to there.
        horizon (int): max number of timesteps to play.
    """

    print(VISION_LIMIT, VISION_BOUND, VISION_LIMIT_AWARE, EXPLORE, agent_unstuck, human_unstuck, SEARCH_DEPTH, KB_SEARCH_DEPTH, KB_UPDATE_DELAY)

    start_time = time.time()
    grid = lvl_str2grid(lvl_str)
    mdp = SteakHouseGridworld.from_grid(grid, CONFIG)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=horizon)

    COUNTERS_PARAMS = {
        'start_orientations': True,
        'wait_allowed': True,
        'counter_goals': [],
        'counter_drop': mdp.terrain_pos_dict['X'],
        'counter_pickup': mdp.terrain_pos_dict['X'],
        'same_motion_goals': True
    }

    # ml_action_manager = planners.MediumLevelActionManager(mdp, NO_COUNTERS_PARAMS)

    # hmlp = planners.HumanMediumLevelPlanner(mdp, ml_action_manager, [0.5, (1.0-0.5)], 0.5)
    # human_agent = agent.biasHumanModel(ml_action_manager, [0.5, (1.0-0.5)], 0.5, auto_unstuck=True)
    mlp = planners.MediumLevelPlanner.from_pickle_or_compute(mdp, COUNTERS_PARAMS, force_compute=True)  
    human_agent = agent.SteakLimitVisionHumanModel(mlp, env.state, auto_unstuck=human_unstuck, explore=EXPLORE, vision_limit=VISION_LIMIT, vision_bound=VISION_BOUND, kb_update_delay=KB_UPDATE_DELAY, debug=True)
    # human_agent = agent.GreedySteakHumanModel(mlp)
    # human_agent = agent.CoupledPlanningAgent(mlp)
    human_agent.set_agent_index(1)

    qmdp_start_time = time.time()
    # mdp_planner = planners.SteakHumanSubtaskQMDPPlanner.from_pickle_or_compute(mdp, COUNTERS_PARAMS, force_compute_all=True, jmp = mlp.ml_action_manager.joint_motion_planner, vision_limited_human=human_agent)
    # mdp_planner = planners.SteakKnowledgeBasePlanner.from_pickle_or_compute(mdp, COUNTERS_PARAMS, force_compute_all=True, jmp = mlp.ml_action_manager.joint_motion_planner, vision_limited_human=human_agent)# if VISION_LIMIT else None)
    # ai_agent = agent.QMDPAgent(mlp, env)
    # ai_agent = agent.GreedySteakHumanModel(mlp)
    
    mdp_planner = None
    if not VISION_LIMIT_AWARE and VISION_LIMIT:
        non_limited_human = agent.SteakLimitVisionHumanModel(mlp, env.state, auto_unstuck=human_unstuck, explore=EXPLORE, vision_limit=False, vision_bound=0, kb_update_delay=1, debug=True)
        non_limited_human.set_agent_index(1)
        mdp_planner = planners.SteakKnowledgeBasePlanner.from_pickle_or_compute(mdp, COUNTERS_PARAMS, force_compute_all=True, jmp = mlp.ml_action_manager.joint_motion_planner, vision_limited_human=non_limited_human, debug=True, search_depth=SEARCH_DEPTH, kb_search_depth=KB_SEARCH_DEPTH)
    else:
        limited_human = agent.SteakLimitVisionHumanModel(mlp, env.state, auto_unstuck=human_unstuck, explore=EXPLORE, vision_limit=VISION_LIMIT, vision_bound=VISION_BOUND, kb_update_delay=KB_UPDATE_DELAY, debug=True)
        limited_human.set_agent_index(1)
        mdp_planner = planners.SteakKnowledgeBasePlanner.from_pickle_or_compute(mdp, COUNTERS_PARAMS, force_compute_all=True, jmp = mlp.ml_action_manager.joint_motion_planner, vision_limited_human=limited_human, debug=False, search_depth=SEARCH_DEPTH, kb_search_depth=KB_SEARCH_DEPTH)
    
    ai_agent = agent.MediumQMdpPlanningAgent(mdp_planner, greedy=True, auto_unstuck=agent_unstuck, low_level_action_flag=True, vision_limit=VISION_LIMIT)
    ai_agent.set_agent_index(0)

    if VISION_LIMIT_AWARE and VISION_LIMIT:
        if agent_save_path is not None:
            with open(agent_save_path, 'wb') as f:
                pickle.dump(ai_agent, f)

    agent_pair = agent.AgentPair(ai_agent, human_agent) # if use QMDP, the first agent has to be the AI agent
    print("It took {} seconds for planning".format(time.time() - start_time))
    game_start_time = time.time()
    s_t, joint_a_t, r_t, done_t = env.run_agents(agent_pair, include_final_state=True, display=True)
    # print("It took {} seconds for qmdp to compute".format(game_start_time - qmdp_start_time))
    # print("It took {} seconds for playing the entire level".format(time.time() - game_start_time))
    # print("It took {} seconds to plan".format(time.time() - start_time))
    trajectory = s_t[:,1][:-1].tolist()

    if VISION_LIMIT_AWARE and VISION_LIMIT:
        if value_kb_save_path is not None:
            with open(value_kb_save_path, 'wb') as f:
                pickle.dump([ai_agent.mdp_planner.world_state_cost_dict, ai_agent.mdp_planner.track_state_kb_map], f)

    del mlp, mdp_planner
    gc.collect()

    return (done_t, trajectory, len(trajectory))


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


def correct_study_type(study_type, lvl_type):
    if study_type == "all" and lvl_type != "trial":
        return True
    else:
        return lvl_type.startswith(study_type)

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
    parser.add_argument(
        '--study',
        help=
        "Which set of study to run. Should be one of 'trial', 'even_workloads', 'uneven_workloads', 'high_team_fluency', 'low_team_fluency' and 'all'.",
        default=False)

    parser.add_argument('--replay',
                        action='store_true',
                        help='Whether use the replay mode',
                        default=False)
    parser.add_argument('--reload',
                        action='store_true',
                        help='Whether to continue running a previous study',
                        default=False)
    parser.add_argument('-l',
                        '--log_index',
                        help='Integer: index of the study log',
                        required=False,
                        default=-1)
    parser.add_argument('-type',
                        help='Integer: type of the game level.',
                        required=False,
                        default=None)
    parser.add_argument('--human_play',
                        action='store_true',
                        help='Whether to continue running a previous study',
                        default=False)
    parser.add_argument('-m',
                        '--human_play_mode',
                        help='Integer: index of the agent look ahead',
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

    np.random.seed(1)
    # not replay, run the study
    if not opt.replay:
        # read in human study levels
        if not opt.human_play:
            study_lvls = None
            if opt.human_play_mode == '0':
                study_lvls = pd.read_csv(
                os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "all.csv"))
            elif opt.human_play_mode == '1':
                study_lvls = pd.read_csv(
                os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "all-2.csv"))
            elif opt.human_play_mode == '2':
                study_lvls = pd.read_csv(
                os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "real_study_lvls.csv"))
            elif opt.human_play_mode == '3':
                study_lvls = pd.read_csv(
                os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "real_user_lvls.csv"))
            elif opt.human_play_mode == '4':
                study_lvls = pd.read_csv(
                os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "real_user_lvls_kb_search0.csv"))
            elif opt.human_play_mode == '5':
                study_lvls = pd.read_csv(
                os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "real_user_lvls_kb_search6.csv"))
            elif opt.human_play_mode == '6':
                study_lvls = pd.read_csv(
                os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "real_user_lvls_kb_search3.csv"))
                
            # study_lvls = pd.read_csv(
            #     os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "new_study_lvls.csv"))
        # else:
        #     study_lvls = pd.read_csv(
        #             os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "real_user_lvls.csv"))
        
        # running a new study
        if not opt.reload:
            # quit if study type not recognized
            if opt.study not in ALL_STUDY_TYPES:
                print(
                    "Study type not supported. Must be one of the following:",
                    ", ".join(ALL_STUDY_TYPES))
                exit(1)

            if opt.study == 'trial':
                study_lvls = pd.read_csv(
                os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "trial_lvls.csv"))
                # lvl_config = study_lvls.iloc[0]

                for index, lvl_config in study_lvls.iterrows():
                    agent_save_path, value_kb_save_path, human_save_path = gen_save_pths(lvl_config)
                    print("trial")
                    print(lvl_config["lvl_str"])
                    if index < 2:
                        human_play(
                            lvl_config["lvl_str"],
                            ai_agent=StayAgent(),
                            human_save_path=human_save_path,
                            trial=True,
                            lvl_config=lvl_config
                        )
                    else:
                        human_play(
                            lvl_config["lvl_str"],
                            human_save_path=human_save_path,
                            agent_save_path=agent_save_path,
                            value_kb_save_path=value_kb_save_path,
                            trial=True,
                            lvl_config=lvl_config,
                        )

            else:
                # initialize the result log files
                human_log_csv = create_human_exp_log()

                if opt.human_play:
                    unaware_lvls = pd.read_csv(
                        os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "user_study_unaware.csv"))
                    aware_lvls = pd.read_csv(
                        os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "user_study_aware.csv"))
                    aware_not_act_lvls = pd.read_csv(
                        os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "user_study_aware_not_act.csv"))
                    
                    if opt.human_play_mode == '0':
                        user_study_lvls = [unaware_lvls, aware_not_act_lvls, aware_lvls]
                    elif opt.human_play_mode == '1':
                        user_study_lvls = [unaware_lvls, aware_lvls, aware_not_act_lvls]

                    for lvls in user_study_lvls:
                        # lvls = study_lvls.sample(frac=1)
                        # play all of the levels
                        for index, lvl_config in lvls.iterrows():
                            agent_save_path, value_kb_save_path, human_save_path = gen_save_pths(lvl_config)
                            print(lvl_config["lvl_type"])
                            results = human_play(lvl_config["lvl_str"],
                                        human_save_path = human_save_path,
                                        agent_save_path=agent_save_path,
                                        value_kb_save_path=value_kb_save_path,
                                        lvl_config=lvl_config)
                            
                            # write the results
                            write_to_human_exp_log(human_log_csv, results,
                                                    lvl_config)

                # shuffle the order if playing all
                # if opt.study == 'all':
                #     study_lvls = study_lvls
                #     # study_lvls = study_lvls.sample(frac=1)

                # # play all of the levels
                # for index, lvl_config in study_lvls.iterrows():
                #     # check study type:
                #     if correct_study_type(opt.study, lvl_config["lvl_type"]):
                #         agent_save_path, value_kb_save_path, human_save_path = gen_save_pths(lvl_config)
                #         print(lvl_config["lvl_type"])
                #         if not opt.human_play:
                #             results = agents_play(lvl_config["lvl_str"],
                #                                 agent_save_path=agent_save_path,
                #                                 value_kb_save_path = value_kb_save_path,
                #                                 VISION_LIMIT = lvl_config["vision_limit"],
                #                                 VISION_BOUND = int(lvl_config["vision_bound"]),
                #                                 VISION_LIMIT_AWARE = lvl_config["vision_limit_aware"],
                #                                 SEARCH_DEPTH= lvl_config["search_depth"],
                #                                 KB_SEARCH_DEPTH= lvl_config["kb_search_depth"],
                #                                 KB_UPDATE_DELAY=lvl_config["kb_update_delay"])
                #         else:
                #             results = human_play(lvl_config["lvl_str"],
                #                         human_save_path = human_save_path,
                #                         agent_save_path=agent_save_path,
                #                         value_kb_save_path=value_kb_save_path,
                #                         lvl_config=lvl_config)
                #         # write the results
                #         if lvl_config["lvl_type"] != "trial":
                #             write_to_human_exp_log(human_log_csv, results,
                #                                    lvl_config)

        # loading an existing study and continue running it.
        else:
            log_index = opt.log_index
            assert int(log_index) >= 0
            human_log_csv, human_log_data = load_human_log_data(log_index)
            
            if opt.human_play_mode == '0':
                study_lvls = pd.read_csv(
                os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "all_user_lvls_0.csv"))
                RELOAD_STUDY_TYPE = RELOAD_STUDY_TYPE0
            elif opt.human_play_mode == '1':
                study_lvls = pd.read_csv(
                os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "all_user_lvls_1.csv"))
                RELOAD_STUDY_TYPE = RELOAD_STUDY_TYPE1

            # find levels need to run and play them
            for lvl_type in RELOAD_STUDY_TYPE:
                if lvl_type not in human_log_data["lvl_type"].to_list():
                    lvl_config = study_lvls[study_lvls["lvl_type"] ==
                                            lvl_type].iloc[0]
                    lvl_str = lvl_config["lvl_str"]
                    print(lvl_config["lvl_type"])
                    print(lvl_str)
                    agent_save_path, value_kb_save_path, human_save_path = gen_save_pths(lvl_config)
                    if not opt.human_play:
                        results = agents_play(lvl_str,
                                        agent_save_path=agent_save_path,
                                        value_kb_save_path=value_kb_save_path,
                                        VISION_LIMIT = lvl_config["vision_limit"],
                                        VISION_BOUND = int(lvl_config["vision_bound"]),
                                        VISION_LIMIT_AWARE = lvl_config["vision_limit_aware"],
                                        SEARCH_DEPTH= lvl_config["search_depth"],
                                        KB_SEARCH_DEPTH= lvl_config["kb_search_depth"],
                                        KB_UPDATE_DELAY=lvl_config["kb_update_delay"])
                    else:
                        results = human_play(lvl_str,
                                        human_save_path=human_save_path,
                                        agent_save_path=agent_save_path,
                                        value_kb_save_path=value_kb_save_path,
                                        lvl_config=lvl_config)
                    # write the results
                    if lvl_config["lvl_type"] != "trial":
                        write_to_human_exp_log(human_log_csv, results,
                                            lvl_config)

    # replay the specified study
    else:
        log_index = opt.log_index
        _, human_log_data = load_human_log_data(log_index)
        NO_FOG_COPY=True

        # shuffle the order if playing all
        if opt.type == 'all':
            # play all of the levels
            for index, lvl_config in human_log_data.iterrows():
                # get level string and logged joint actions from log file
                lvl_str = lvl_config["lvl_str"]
                joint_actions = ast.literal_eval(lvl_config["joint_actions"])

                subtask_log = ast.literal_eval(lvl_config["subtask_log"])
                human_kb_log = ast.literal_eval(lvl_config["human_kb_log"])
                world_kb_log = ast.literal_eval(lvl_config["world_kb_log"])

                if opt.gen_plot:
                    plot_kb_and_subtasks(joint_actions, subtask_log, human_kb_log, world_kb_log, log_dir=os.path.join(LSI_STEAK_STUDY_RESULT_DIR, log_index), log_name=lvl_config["lvl_type"])
                    
                # replay the game
                if opt.gen_vid:
                    replay_with_joint_actions(lvl_str, joint_actions, log_dir=os.path.join(LSI_STEAK_STUDY_RESULT_DIR, log_index), log_name=lvl_config["lvl_type"], view_angle=lvl_config["vision_bound"])
        
                    if NO_FOG_COPY and lvl_config["vision_bound"] > 0:
                        replay_with_joint_actions(lvl_str, joint_actions, log_dir=os.path.join(LSI_STEAK_STUDY_RESULT_DIR, log_index), log_name=lvl_config["lvl_type"]+'_nofog', view_angle=0)


        else:
            # get level string and logged joint actions from log file
            lvl_str = human_log_data[human_log_data["lvl_type"] ==
                                    opt.type]["lvl_str"].iloc[0]
            vision_bound = human_log_data[human_log_data["lvl_type"] ==
                                    opt.type]["vision_bound"].iloc[0]
            lvl_type = human_log_data[human_log_data["lvl_type"] ==
                                    opt.type]["lvl_type"].iloc[0]
            joint_actions = ast.literal_eval(human_log_data[
                human_log_data["lvl_type"] == opt.type]["joint_actions"].iloc[0])

            subtask_log = ast.literal_eval(human_log_data[
                human_log_data["lvl_type"] == opt.type]["subtask_log"].iloc[0])
            human_kb_log = ast.literal_eval(human_log_data[
                human_log_data["lvl_type"] == opt.type]["human_kb_log"].iloc[0])
            world_kb_log = ast.literal_eval(human_log_data[
                human_log_data["lvl_type"] == opt.type]["world_kb_log"].iloc[0])

            if opt.gen_plot:
                plot_kb_and_subtasks(joint_actions, subtask_log, human_kb_log, world_kb_log, log_dir=os.path.join(LSI_STEAK_STUDY_RESULT_DIR, log_index), log_name=lvl_type)

            # replay the game
            if opt.gen_vid:
                replay_with_joint_actions(lvl_str, joint_actions, log_dir=os.path.join(LSI_STEAK_STUDY_RESULT_DIR, log_index), log_name=lvl_type, view_angle=vision_bound)

                if NO_FOG_COPY and vision_bound > 0:
                    replay_with_joint_actions(lvl_str, joint_actions, log_dir=os.path.join(LSI_STEAK_STUDY_RESULT_DIR, log_index), log_name=lvl_type+'_nofog', view_angle=0)

