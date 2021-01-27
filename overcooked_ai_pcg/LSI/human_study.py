import pygame
import os
import csv
import ast
import time
import toml
import argparse
import pickle
import numpy as np
import pandas as pd
from pprint import pprint
from overcooked_ai_pcg import (TEAM_FLUENCY_DIR, HIGH_TEAM_FLUENCY_DIR,
                               WORKLOADS_DIR, LOW_TEAM_FLUENCY_DIR,
                               EVEN_WORKLOADS_DIR, UNEVEN_WORKLOADS_DIR,
                               TRIAL_DIR, LSI_HUMAN_STUDY_RESULT_DIR)
from overcooked_ai_pcg.helper import init_env, init_qmdp_agent
from overcooked_ai_py.agents.agent import HumanPlayer
from overcooked_ai_py.mdp.overcooked_mdp import Direction, Action

HUMAN_STUDY_ENV_HORIZON = 150

ALL_STUDY_TYPES = [
    'even-workloads',
    'uneven-workloads',
    'high-team_fluency',
    'low-team_fluency',
]


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
    def __init__(self, env, agent, agent_idx, rand_seed, slow_time=False):
        self._running = True
        self._display_surf = None
        self.env = env
        self.agent = agent
        self.agent_idx = agent_idx
        self.human_player = HumanPlayer()
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
        self.human_player.set_agent_index(1 - self.agent_idx)
        self.human_player.set_mdp(self.env.mdp)

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

        # update logs of human player for bc calculations
        self.human_player.update_logs(next_state, my_action)

        if timestep_sparse_reward > 0:
            self.checkpoints[self.cur_order] = self.timestep
            self.cur_order += 1

        self.total_sparse_reward += timestep_sparse_reward
        self.timestep += 1
        print("Timestep:", self.timestep)
        return done, next_state

    def on_loop(self):
        pass

    def on_render(self):
        self.env.render()

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
        workloads = self.last_state.get_player_workload()
        concurr_active = self.last_state.cal_concurrent_active_sum()
        stuck_time = self.last_state.cal_total_stuck_time()

        fitness = self.total_sparse_reward + 1
        for checked_time in reversed(self.checkpoints):
            fitness *= self.env.horizon
            fitness -= checked_time

        return (fitness, self.total_sparse_reward, self.checkpoints, workloads,
                self.joint_actions, concurr_active, stuck_time)


def load_qmdp_agent(env, agent_save_path):
    ai_agent = None
    if agent_save_path is not None:
        # agent saved before, load it.
        if os.path.exists(agent_save_path):
            with open(agent_save_path, 'rb') as f:
                ai_agent = pickle.load(f)

    # agent not found, recreate it and save it if a path is given.
    if ai_agent == None:
        ai_agent = init_qmdp_agent(env.mdp)
        if agent_save_path is not None:
            with open(agent_save_path, 'wb') as f:
                pickle.dump(ai_agent, f)
    return ai_agent


def replay_with_joint_actions(lvl_str, joint_actions, horizon=100):
    """Replay a game play with given level and joint actions.

    Args:
        joint_actions (list of tuple of tuple): Joint actions.
    """
    env = init_env(lvl_str, horizon=HUMAN_STUDY_ENV_HORIZON)
    done = False
    i = 0
    while not done:
        env.render()
        next_state, timestep_sparse_reward, done, info = env.step(
            joint_actions[i])
        i += 1
        time.sleep(0.2)


def human_play(
    lvl_str,
    ai_agent=None,
    agent_save_path=None,
):
    """Function that allows human to play with an ai_agent.

    Args:
        lvl_str (str): Level string.
        agent_save_path (str): Path to the pre-saved ai agent. If nothing is
            found, it will be saved to there.
        ai_agent (Agent): Agent that human plays with. Default is QMDP agent.
    """
    env = init_env(lvl_str, horizon=HUMAN_STUDY_ENV_HORIZON)
    ai_agent = load_qmdp_agent(env, agent_save_path)
    theApp = OvercookedGame(env, ai_agent, agent_idx=0, rand_seed=10)
    return theApp.on_execute()


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


def write_to_human_exp_log(human_log_csv, lvl_type_full, results, lvl_config):
    """Write to human exp log.

    Args:
        human_log_csv (str): Path to the human_log.csv file.
        lvl_type_full (str): Type of the level. Format {type}_{bc_type}-{idx}.
            e.g. low_team_fluency-2, even_workload-1
        results (tuple) all of the results returned from the human study.
        lvl_config (dic): toml config dic of the level.
    """
    assert os.path.exists(human_log_csv)

    to_write = [
        lvl_type_full,
        lvl_config["ID"] if "ID" in lvl_config else None,
        lvl_config["exp_log_dir"] if "exp_log_dir" in lvl_config else None,
        lvl_config["row_index"] if "row_index" in lvl_config else None,
        lvl_config["column_index"] if "column_index" in lvl_config else None,
        lvl_config["matrix_index"] if "matrix_index" in lvl_config else None,
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
    while os.path.isdir(os.path.join(LSI_HUMAN_STUDY_RESULT_DIR,
                                     str(exp_dir))):
        exp_dir += 1
    exp_dir = os.path.join(LSI_HUMAN_STUDY_RESULT_DIR, str(exp_dir))
    os.mkdir(exp_dir)

    # create csv file to store results
    human_log_csv = os.path.join(exp_dir, 'human_log.csv')

    # construct labels
    data_labels = [
        "lvl_type",
        "ID",
        "exp_log_dir",  # exp log directory which determines elite map
        "row_index",
        "column_index",
        "matrix_index",  # could be None cuz elite map can be 2D or 3D
        "fitness",
        "total_sparse_reward",
        "checkpoints",
        "workloads",
        "joint_actions",
        "concurr_active",
        "stuck_time",
        "lvl_str",
    ]

    write_row(human_log_csv, data_labels)

    return human_log_csv


def questionaire():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--study',
        help=
        "Which set of study to run. Should be one of 'even-workloads', 'uneven-workloads', 'high-team_fluency', 'low-team_fluency' and 'all'.",
        default=False)

    parser.add_argument('--replay',
                        action='store_true',
                        help='Whether use the replay mode',
                        default=False)
    parser.add_argument('-l',
                        '--log_index',
                        help='Integer: index of the study log',
                        required=False,
                        default=-1)
    parser.add_argument('-id',
                        help='Integer: ID of the game level.',
                        required=False,
                        default=-1)
    opt = parser.parse_args()

    # not replay, run the study
    if not opt.replay:
        # initialize the result log files
        human_log_csv = create_human_exp_log()

        # read in human study levels
        study_lvls = {"workloads": {}, "team_fluency": {}, "trial": {}}

        study_lvls["trial"]["trial"] = read_in_study_lvl(TRIAL_DIR)
        study_lvls["workloads"]["even"] = read_in_study_lvl(EVEN_WORKLOADS_DIR)
        study_lvls["workloads"]["uneven"] = read_in_study_lvl(
            UNEVEN_WORKLOADS_DIR)
        study_lvls["team_fluency"]["low"] = read_in_study_lvl(
            LOW_TEAM_FLUENCY_DIR)
        study_lvls["team_fluency"]["high"] = read_in_study_lvl(
            HIGH_TEAM_FLUENCY_DIR)

        # construct the config file based on study mode
        study_configs = [
            {
                "lvl_types": ["trial"],
                "exp_type": "trial",
                "dir": TRIAL_DIR,
            },
        ]
        if opt.study == "all":
            study_configs += [
                {
                    "lvl_types": ["even", "uneven"],
                    "exp_type": "workloads",
                    "dir": WORKLOADS_DIR,
                },
                {
                    "lvl_types": ["high", "low"],
                    "exp_type": "team_fluency",
                    "dir": TEAM_FLUENCY_DIR,
                },
            ]
            # while playing all levels, semi-randomize the order
            # this shuffles each 'lvl_types' array in place. Note that it would
            # shuffle the array in place so we don't have to assign it again.
            [np.random.shuffle(x["lvl_types"]) for x in study_configs]
        elif opt.study in ALL_STUDY_TYPES:
            # get level type (high/low, even/uneven)
            # and experiment type workloads/team_fluency
            lvl_type, exp_type = opt.study.split('-')
            if exp_type == 'team_fluency':
                _dir = TEAM_FLUENCY_DIR
            elif exp_type == 'workloads':
                _dir = WORKLOADS_DIR

            study_configs += [
                {
                    "lvl_types": [lvl_type],
                    "exp_type": exp_type,
                    "dir": _dir,
                },
            ]
        else:
            print("Study type not supported.")
            exit(1)

        # run the study
        for study_config in study_configs:
            lvl_types = study_config["lvl_types"]
            exp_type = study_config["exp_type"]
            _dir = study_config["dir"]
            for lvl_type in lvl_types:
                # all levels to play
                to_plays = study_lvls[exp_type][lvl_type]
                for i in range(len(to_plays)):
                    lvl_str = to_plays[i]["lvl_str"]
                    # path to which the agent pkl is stored
                    if exp_type == "trial":
                        agent_save_path = os.path.join(_dir, f"agent{i}.pkl")
                    else:
                        agent_save_path = os.path.join(
                            os.path.join(_dir, lvl_type), f"agent{i}.pkl")
                    # let the human play the level
                    results = human_play(lvl_str,
                                         agent_save_path=agent_save_path)
                    lvl_type_full = f"{lvl_type}_{exp_type}-{i}"
                    # write the results
                    if exp_type != "trial":
                        write_to_human_exp_log(human_log_csv, lvl_type_full,
                                               results, to_plays[i])

                # TODO: implement questionaire after each type of level is
                # finished
                questionaire()

    # replay the specified study
    else:
        log_index = opt.log_index
        lvl_id = int(opt.id)
        assert int(log_index) >= 0
        assert lvl_id >= 0

        # get level string and logged joint actions from log file
        human_log_csv = os.path.join(LSI_HUMAN_STUDY_RESULT_DIR, log_index,
                                     "human_log.csv")
        human_log_data = pd.read_csv(human_log_csv)
        lvl_str = human_log_data[human_log_data["ID"] == lvl_id]["lvl_str"][0]
        joint_actions = ast.literal_eval(
            human_log_data[human_log_data["ID"] == lvl_id]["joint_actions"][0])

        # replay the game
        replay_with_joint_actions(lvl_str,
                                  joint_actions,
                                  horizon=HUMAN_STUDY_ENV_HORIZON)
