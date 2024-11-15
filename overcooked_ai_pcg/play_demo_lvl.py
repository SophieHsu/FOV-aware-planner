import pygame
import os
import csv
import time
import json
import toml
import argparse
import pandas as pd
import pickle
import numpy as np
from overcooked_ai_pcg import LSI_CONFIG_EXP_DIR, LSI_LOG_DIR, LSI_CONFIG_ALGO_DIR, LSI_CONFIG_MAP_DIR, LSI_CONFIG_AGENT_DIR
from overcooked_ai_pcg.helper import read_in_lsi_config, init_env_and_agent
from overcooked_ai_pcg.LSI.qd_algorithms import Individual
from overcooked_ai_py.mdp.overcooked_mdp import Direction, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from overcooked_ai_py.agents.agent import StayAgent, RandomAgent, AgentFromPolicy, GreedyHumanModel

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Direction, Action


class App:
    """Class to run an Overcooked Gridworld game, leaving one of the players as
       fixed.
       Useful for debugging. Most of the code from http://pygametutorials.wikidot.com/tutorials-basic.

    Args:
        env: Overcooked Environment.
        agent: AI agent that the human plays game with.
        rand_seed: random seed.
        agent_idx: index of the AI agent.
        slow_time (bool): whether the AI agent take multiple actions each time
            the human take one action.
    """
    def __init__(self, env, agent, agent_idx, rand_seed, slow_time=False):
        self._running = True
        self._display_surf = None
        self.env = env
        self.agent = agent
        self.agent_idx = agent_idx
        self.slow_time = slow_time
        self.total_sparse_reward = 0
        self.last_state = None
        self.rand_seed = rand_seed

        # Saves when each soup (order) was delivered
        self.checkpoints = [env.horizon - 1] * env.num_orders
        self.cur_order = 0
        self.timestep = 0
        self.joint_actions = []

    def on_init(self):
        pygame.init()

        # Adding AI agent as teammate
        self.agent.set_agent_index(self.agent_idx)
        self.agent.set_mdp(self.env.mdp)

        self.env.render()
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

        if event.type == pygame.QUIT or done:
            self._running = False
            self.last_state = next_state

    def step_env(self, my_action):
        agent_action = self.agent.action(self.env.state)[0]

        if self.agent_idx == 1:
            joint_action = (my_action, agent_action)
        else:
            joint_action = (agent_action, my_action)

        self.joint_actions.append(joint_action)
        next_state, timestep_sparse_reward, done, info = self.env.step(
            joint_action)

        if timestep_sparse_reward > 0:
            self.checkpoints[self.cur_order] = self.timestep
            self.cur_order += 1

        self.total_sparse_reward += timestep_sparse_reward
        self.timestep += 1
        self.env.render()
        return done, next_state

    def on_loop(self):
        pass

    def on_render(self):
        pass

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while (self._running):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()

        workloads = self.last_state.get_player_workload()
        concurr_active = self.last_state.cal_concurrent_active_sum()
        stuck_time = self.last_state.cal_total_stuck_time()

        fitness = self.total_sparse_reward + 1
        for checked_time in reversed(self.checkpoints):
            fitness *= self.env.horizon
            fitness -= checked_time

        return fitness, self.total_sparse_reward, self.checkpoints, workloads, self.joint_actions, concurr_active, stuck_time


class GameConfigLog:
    def __init__(self, bc_exp_dir, filename):
        self.db = {}
        self.full_path = os.path.join(bc_exp_dir, filename + '.pkl')

    def addData(self, keys, values):
        for key, value in zip(keys, values):
            self.db[key] = value

    def storeData(self):
        dbfile = open(self.full_path, 'wb')
        pickle.dump(self.db, dbfile)
        dbfile.close()

    def loadData(self):
        # for reading also binary mode is important
        dbfile = open(self.full_path, 'rb')
        self.db = pickle.load(dbfile)
        agent1 = self.db['agent1']
        agent2 = self.db['agent2']
        mdp = self.db['mdp']
        dbfile.close()

        return agent1, agent2, mdp

    def is_exist(self):
        return os.path.exists(self.full_path)


def write_row(log_dir, to_add):
    """Append a row to csv file"""
    print(os.path.join(log_dir, 'human_exp_log.csv'))
    from IPython import embed
    embed
    with open(os.path.join(log_dir,'human_exp_log.csv'), 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(to_add)
        f.close()


def init_log(log_dir):
    # remove the file if exists
    if os.path.exists(os.path.join(log_dir, 'human_exp_log.csv')):
        os.remove(os.path.join(log_dir, 'human_exp_log.csv'))

    # construct labels
    data_labels = [
        "ID", "feature 1", "feature 2", "row index", "column index",
        "joint actions", "fitness"
    ]
    # We need to be told how many orders we have
    data_labels += [
        "scores", "order_delivered", "player_workloads", "concurr_active",
        "stuck_time", "random seed", "lvl_str"
    ]
    write_row(log_dir, data_labels)


def gen_log_file_name(agent_config, f1, f2, row_idx, col_idx, ind_id):
    agent1_config = agent_config["Agent1"]
    agent2_config = agent_config["Agent2"]

    #log_file = "human_exp/human_w_"
    log_file = "human_exp/qmdp_"

    if agent1_config["name"] == "fixed_plan_agent" and agent2_config[
            "name"] == "fixed_plan_agent":
        log_file += "fixed_plan"
    elif agent1_config["name"] == "preferenced_human" and agent2_config[
            "name"] == "human_aware_agent":
        log_file += "human_aware"
    elif agent1_config["name"] == "qmdp_agent" and agent2_config[
            "name"] == "greedy_agent":
        log_file += "qmdp"
    elif agent1_config["name"] == "mdp_agent" and agent2_config[
            "name"] == "greedy_agent":
        log_file += "mdp"
    else:
        print("Agent configs not recognized. Please check your config file.")

    log_file += ("_joint_actions_" + str(f1) + "_" + str(f2) + "_" +
                 str(row_idx) + "_" + str(col_idx) + "_" + str(ind_id))

    return log_file


def log_actions(ind, agent_config, log_dir, individuals, f1, f2, row_idx,
                col_idx, ind_id, log_file_name):

    full_path = os.path.join(log_dir, log_file_name + '.json')

    # log the joint actions if not logged before
    with open(full_path, "w") as f:
        json.dump(
            {
                "joint_actions": ind.joint_actions,
                "lvl_str": ind.level,
                "fitness": ind.fitness,
                "score": ind.scores,
                "checkpoints": ind.checkpoints,
                "player_workloads": ind.player_workloads,
                "concurr_active": int(ind.concurr_active),
                "stuck_time": int(ind.stuck_time),
                "rand_seed": int(ind.rand_seed),
            }, f)
        print("Joint actions saved")

    write_row(log_dir, [
        ind_id, f1, f2, row_idx, col_idx, ind.joint_actions, ind.fitness,
        ind.scores, *ind.checkpoints, ind.player_workloads, ind.concurr_active,
        ind.stuck_time, ind.rand_seed, ind.level
    ])

    write_row(log_dir, [
        "", "", "", "", "", "", individuals["fitness"][ind_id],
        individuals["score_preferenced_human_w_human_aware_agent"][ind_id],
        individuals["order_delivered_preferenced_human_w_human_aware_agent"]
        [ind_id],
        individuals["player_workload_preferenced_human_w_human_aware_agent"]
        [ind_id], individuals["cc_active"][ind_id],
        individuals["stuck_time"][ind_id]
    ])


def play(agent_configs, individuals, f1, f2, row_idx, col_idx, bc_exp_dir,
         curr_exp_log_dir, log_file_name, ind_id):
    """
    Find the individual in the specified cell in the elite map
    and run overcooked game with the specified agents

    Args:
        elite_map (list): list of logged cell strings.
                          See elite_map.csv for detail.
        agent_configs: toml config object of agents
        individuals (pd.dataFrame): all individuals logged
        f1, f2 (int, int): index of the features to use
        row_idx, col_idx (int, int): index of the cell in the elite map
    """
    ind_id = int(ind_id)
    lvl_str = individuals["lvl_str"][ind_id]
    print("Playing in individual %d" % ind_id)
    print(lvl_str)
    ind = Individual()
    ind.level = lvl_str
    ind.human_preference = individuals["human_preference"][ind_id]
    ind.human_adaptiveness = individuals["human_adaptiveness"][ind_id]
    #ind.rand_seed = individuals["rand_seed"][ind_id]
    ind.rand_seed = 0
    
    agent1 = None; agent2 = None; env = None
    game_log = GameConfigLog(bc_exp_dir, log_file_name)
    if game_log.is_exist():
        agent1, agent2, mdp = game_log.loadData()
        env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=100)
    else:
        agent1, agent2, env, mdp = init_env_and_agent(ind, agent_configs[-1])
        game_log.addData(["agent1", "agent2", "mdp"], [agent1, agent2, mdp])
        game_log.storeData()

    # assert type(agent1) == 

    theApp = App(env,
                 agent1,
                 rand_seed=ind.rand_seed,
                 agent_idx=0,
                 slow_time=False)
    ind.fitness, ind.scores, ind.checkpoints, ind.player_workloads, ind.joint_actions, ind.concurr_active, ind.stuck_time = theApp.on_execute(
    )

    print("Fitness: {}; Total sparse reward: {};".format(
        ind.fitness, ind.scores))
    print("Checkpoints", ind.checkpoints)
    print("Workloads:", ind.player_workloads, "; Concurrently active:",
          ind.concurr_active, "; Stuck time:", ind.stuck_time)

    log_actions(ind, agent_configs[-1], curr_exp_log_dir, individuals, f1, f2,
                row_idx, col_idx, ind_id, log_file_name)


def create_human_exp_dir(extreme_bcs_dir, bc_exps):
    """ Create human_exp/<exp_id> under log_dir/extreme_bcs. <exp_id> would be
        determined by the first digit that does not exist under the human_exp
        directory.

        Returns:
            Directory path to the exp result log folder.
    """
    human_exp_dir = os.path.join(extreme_bcs_dir, "human_exp_results")
    if not os.path.isdir(human_exp_dir):
        os.mkdir(human_exp_dir)

    # increment from 0 to find a directory name that does not exist yet.
    exp_dir = 0
    while os.path.isdir(os.path.join(human_exp_dir, str(exp_dir))):
        exp_dir += 1
    exp_dir = os.path.join(human_exp_dir, str(exp_dir))
    os.mkdir(exp_dir)

    # create sub_dirs to store results for different bcs
    for bc_exp in bc_exps:
        exp_sub_dir = os.path.join(exp_dir, bc_exp)
        if not os.path.isdir(exp_sub_dir):
            os.mkdir(exp_sub_dir)
    return exp_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        help='path of experiment config file',
                        required=True)
    parser.add_argument('-l',
                        '--log_dir',
                        help='path of log directory',
                        required=True)
    parser.add_argument('-n',
                        '--num_lvls',
                        help='number of levels to play for each category',
                        required=False,
                        type=int,
                        choices=range(1, 6),
                        default=3)

    opt = parser.parse_args()

    # read in experiment config
    log_dir = opt.log_dir
    exp_bcs_config = toml.load(
        os.path.join(log_dir, "extreme_bcs/exp_bc_config.tml"))

    # read in individuals
    individuals = pd.read_csv(os.path.join(log_dir, "individuals_log.csv"))
    print(individuals['lvl_str'])

    # read in configs
    _, _, elite_map_config, agent_configs = read_in_lsi_config(opt.config)
    bc_exps = exp_bcs_config["demo_bcs"]["bc_names"]

    # write all results to here
    extreme_bcs_dir = os.path.join(log_dir, "extreme_bcs")
    exp_log_dir = create_human_exp_dir(extreme_bcs_dir, bc_exps)

    for bc_exp in bc_exps:
        curr_exp_log_dir = os.path.join(exp_log_dir, bc_exp)
        bc_exp_dir = os.path.join(extreme_bcs_dir, bc_exp)
        init_log(curr_exp_log_dir)
        exp_names = []
        for i, file in enumerate(os.listdir(bc_exp_dir)):
            if file.endswith(".json"):
                exp_names.append(os.path.basename(file)[:-5])

        for exp_name in exp_names[:opt.num_lvls]:
            exp_config = exp_name.split('_')
            f1 = int(exp_config[4])
            f2 = int(exp_config[5])
            row_idx = int(exp_config[6])
            col_idx = int(exp_config[7])
            ind_id = int(exp_config[8])
            log_file_name = gen_log_file_name(agent_configs[-1], f1, f2,
                                              row_idx, col_idx, ind_id)
            # play(agent_configs, individuals, f1, f2, row_idx, col_idx,
            #      bc_exp_dir, curr_exp_log_dir, log_file_name, ind_id)