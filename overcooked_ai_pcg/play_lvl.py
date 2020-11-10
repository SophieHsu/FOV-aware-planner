import pygame
import os
import csv
import time
import json
import toml
import argparse
import pandas as pd
import numpy as np
from overcooked_ai_pcg import LSI_CONFIG_EXP_DIR, LSI_LOG_DIR, LSI_CONFIG_ALGO_DIR, LSI_CONFIG_MAP_DIR, LSI_CONFIG_AGENT_DIR
from overcooked_ai_pcg.helper import read_in_lsi_config, init_env_and_agent
from overcooked_ai_pcg.LSI.qd_algorithms import Individual
from overcooked_ai_py.mdp.overcooked_mdp import Direction, Action


class App:
    """Class to run an Overcooked Gridworld game, leaving one of the players as fixed.
    Useful for debugging. Most of the code from http://pygametutorials.wikidot.com/tutorials-basic."""
    def __init__(self, env, agent, agent2, rand_seed, player_idx, slow_time):
        self._running = True
        self._display_surf = None
        self.env = env
        self.agent = agent
        self.agent2 = agent2
        self.agent_idx = player_idx
        self.slow_time = slow_time
        self.total_sparse_reward = 0
        self.last_state = None
        self.rand_seed = rand_seed

        # Saves when each soup (order) was delivered
        self.checkpoints = [env.horizon - 1] * env.num_orders
        self.cur_order = 0
        self.timestep = 0
        self.joint_actions = []
        # print("Human player index:", player_idx)

    def on_init(self):
        pygame.init()

        # Adding pre-trained agent as teammate
        self.agent.set_agent_index(self.agent_idx)
        self.agent.set_mdp(self.env.mdp)
        self.agent2.set_agent_index(self.agent_idx+1)
        self.agent2.set_mdp(self.env.mdp)

        # print(self.env)
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

            if action in Action.ALL_ACTIONS:

                done, next_state = self.step_env(action)

                if self.slow_time and not done:
                    for _ in range(2):
                        action = Action.STAY
                        done, next_state = self.step_env(action)
                        if done:
                            break

        if event.type == pygame.QUIT or done:
            # print("TOT rew", self.env.cumulative_sparse_rewards)
            self._running = False
            self.last_state = next_state


    def step_env(self, my_action):
        agent_action = self.agent.action(self.env.state)[0]
        agent2_action = self.agent2.action(self.env.state)[0]

        if self.agent_idx == 0:
            joint_action = (my_action, agent2_action)
        else:
            joint_action = (agent2_action, my_action)

        self.joint_actions.append(joint_action)
        next_state, timestep_sparse_reward, done, info = self.env.step(joint_action)

        if timestep_sparse_reward > 0:
            self.checkpoints[self.cur_order] = self.timestep
            self.cur_order += 1

        self.total_sparse_reward += timestep_sparse_reward
        self.timestep += 1

        # print(self.env)
        self.env.render()
        # print("Curr reward: (sparse)", timestep_sparse_reward, "\t(dense)", info["shaped_r_by_agent"])
        # print(self.env.t)
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

        while( self._running ):
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

def log_actions(ind, agent_config, log_dir, f1, f2, row_idx, col_idx, ind_id):
    agent1_config = agent_config["Agent1"]
    agent2_config = agent_config["Agent2"]

    log_file = "human_w_"

    if agent1_config["name"] == "fixed_plan_agent" and agent2_config[
            "name"] == "fixed_plan_agent":
        log_file += "fixed_plan"

    elif agent1_config["name"] == "preferenced_human" and agent2_config[
            "name"] == "human_aware_agent":
        log_file += "human_aware"

    log_file += ("_joint_actions_"+str(f1)+"_"+str(f2)+"_"+str(row_idx)+"_"+str(col_idx)+"_"+str(ind_id)+".json")
    full_path = os.path.join(log_dir, log_file)
    # if os.path.exists(full_path):
    #     print("Joint actions logged before, skipping...")
    #     return

    # log the joint actions if not logged before
    with open(full_path, "w") as f:
        json.dump({
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


def play(elite_map, agent_configs, individuals, f1, f2, row_idx, col_idx,
         log_dir):
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
    for elite in elite_map:
        splited = elite.split(":")
        curr_row_idx = int(splited[f1])
        curr_col_idx = int(splited[f2])
        if curr_row_idx == row_idx and curr_col_idx == col_idx:
            ind_id = int(splited[num_features])
            lvl_str = individuals["lvl_str"][ind_id]
            print("Playing in individual %d" % ind_id)
            print(lvl_str)
            ind = Individual()
            ind.level = lvl_str
            ind.human_preference = individuals["human_preference"][ind_id]
            ind.human_adaptiveness = individuals["human_adaptiveness"][ind_id]
            ind.rand_seed = individuals["rand_seed"][ind_id]

            agent1, agent2, env = init_env_and_agent(ind, agent_configs[-1])

            theApp = App(env, agent1, agent2, rand_seed=ind.rand_seed, player_idx=0, slow_time=False)
            ind.fitness, ind.scores, ind.checkpoints, ind.player_workloads, ind.joint_actions, ind.concurr_active, ind.stuck_time = theApp.on_execute()

            print("Fitness: {}; Total sparse reward: {};".format(ind.fitness, ind.scores))
            print("Checkpoints", ind.checkpoints)
            print("Workloads:", ind.player_workloads, "; Concurrently active:", ind.concurr_active, "; Stuck time:", ind.stuck_time)

            log_actions(ind, agent_configs[-1], log_dir, f1, f2, row_idx, col_idx, ind_id)

            return

    print("No individual found in the specified cell")


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
    parser.add_argument('-row',
                        '--row_idx',
                        help='index f1 in elite map',
                        required=True)
    parser.add_argument('-col',
                        '--col_idx',
                        help='index f2 in elite map',
                        required=True)
    parser.add_argument('-f1',
                        '--feature1_idx',
                        help='index of the first feature to be used',
                        required=False,
                        default=0)
    parser.add_argument('-f2',
                        '--feature2_idx',
                        help='index of the second feature to be used',
                        required=False,
                        default=1)

    opt = parser.parse_args()

    # read in full elite map
    log_dir = opt.log_dir
    elite_map_log_file = os.path.join(log_dir, "elite_map.csv")
    elite_map_log = open(elite_map_log_file, 'r')
    all_rows = list(csv.reader(elite_map_log, delimiter=','))
    elite_map = all_rows[-1][1:]
    elite_map_log.close()

    # read in individuals
    individuals = pd.read_csv(os.path.join(log_dir, "individuals_log.csv"))

    # read in configs
    _, _, elite_map_config, agent_configs = read_in_lsi_config(opt.config)

    # read in feature index
    features = elite_map_config['Map']['Features']
    num_features = len(features)
    f1 = int(opt.feature1_idx)
    f2 = int(opt.feature2_idx)
    assert (f1 < num_features)
    assert (f2 < num_features)

    # read in row/col index
    num_row = features[f1]['resolution']
    num_col = features[f2]['resolution']
    row_idx = int(opt.row_idx)
    col_idx = int(opt.col_idx)
    assert (row_idx < num_row)
    assert (col_idx < num_col)

    play(elite_map, agent_configs, individuals, f1, f2, row_idx, col_idx,
         log_dir)
