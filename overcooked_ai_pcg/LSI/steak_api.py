import ast
import json
import os
from types import SimpleNamespace
from typing import Union
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
import numpy as np
from overcooked_ai_pcg import LSI_STEAK_STUDY_AGENT_DIR, LSI_STEAK_STUDY_CONFIG_DIR, LSI_STEAK_STUDY_RESULT_DIR

from overcooked_ai_pcg.LSI.steak_study import ALL_STUDY_TYPES, DETAILED_STUDY_TYPES, NON_TRIAL_STUDY_TYPES, agents_play, correct_study_type, create_human_exp_log, human_play, human_play_obj, load_human_log_data, replay_with_joint_actions, write_to_human_exp_log
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, PlayerState

app = FastAPI()
overcooked_game = None

import socket
import pygame

pygame.init()

class UdpToPygame():

    def __init__(self):
        UDP_IP="127.0.0.1"
        UDP_PORT=15006
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(0)
        self.sock.bind((UDP_IP,UDP_PORT))

    def update(self):
        try:
            data, addr = self.sock.recvfrom(1024)
            ev = pygame.event.Event(pygame.USEREVENT, {'data': data, 'addr': addr})
            pygame.event.post(ev)
        except socket.error:
            pass 

@app.get("/")
def read_root():
    # done, next_state = overcooked_game.step_env((0,1))
    return {"Hello": "World"}



@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

class Opt():
    def __init__(self, replay=False, reload=False, log_index=-1, type=None, human_play=True, study='all') -> None:
        self.replay = replay
        self.reload = reload
        self.log_index=log_index
        self.type = type
        self.human_play = human_play
        self.study=study

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--study',
    #     help=
    #     "Which set of study to run. Should be one of 'trial', 'even_workloads', 'uneven_workloads', 'high_team_fluency', 'low_team_fluency' and 'all'.",
    #     default=False)
    
    # parser.add_argument('--replay',
    #                     action='store_true',
    #                     help='Whether use the replay mode',
    #                     default=False)
    # parser.add_argument('--reload',
    #                     action='store_true',
    #                     help='Whether to continue running a previous study',
    #                     default=False)
    # parser.add_argument('-l',
    #                     '--log_index',
    #                     help='Integer: index of the study log',
    #                     required=False,
    #                     default=-1)
    # parser.add_argument('-type',
    #                     help='Integer: type of the game level.',
    #                     required=False,
    #                     default=None)
    # parser.add_argument('--human_play',
    #                     action='store_true',
    #                     help='Whether to continue running a previous study',
    #                     default=False)
    # opt = parser.parse_args()
    opt = Opt()

    np.random.seed(1)
    # not replay, run the study
    if not opt.replay:
        # read in human study levels
        if not opt.human_play:
            study_lvls = pd.read_csv(
                os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "study_lvls.csv"))

            # study_lvls = pd.read_csv(
            #     os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "new_study_lvls.csv"))
        else:
            study_lvls = pd.read_csv(
                os.path.join(LSI_STEAK_STUDY_CONFIG_DIR, "human_study_lvls.csv"))
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
                lvl_config = study_lvls.iloc[0]
                agent_save_path = os.path.join(
                    LSI_STEAK_STUDY_AGENT_DIR,
                    "{lvl_type}_{lvl_vision}_{lvl_robot_aware}_{lvl_search_depth}_{lvl_kb_depth}.pkl".format(lvl_type=lvl_config["lvl_type"], lvl_vision=lvl_config["vision_bound"], lvl_robot_aware=lvl_config["vision_limit_aware"], lvl_search_depth=lvl_config["search_depth"], lvl_kb_depth=lvl_config["kb_search_depth"]))
                value_kb_save_path = os.path.join(
                    LSI_STEAK_STUDY_AGENT_DIR,
                    "{lvl_type}_{lvl_vision}_{lvl_robot_aware}_{lvl_search_depth}_{lvl_kb_depth}_v_and_kb.pkl".format(lvl_type=lvl_config["lvl_type"], lvl_vision=lvl_config["vision_bound"], lvl_robot_aware=lvl_config["vision_limit_aware"], lvl_search_depth=lvl_config["search_depth"], lvl_kb_depth=lvl_config["kb_search_depth"]))
                print("trial")
                print(lvl_config["lvl_str"])

                for index, lvl_config in study_lvls.iterrows():
                    human_play(
                        lvl_config["lvl_str"],
                        ai_agent=StayAgent(),
                        agent_save_path=agent_save_path,
                        value_kb_save_path=value_kb_save_path,
                    )

            elif opt.study in NON_TRIAL_STUDY_TYPES:
                # initialize the result log files
                human_log_csv = create_human_exp_log()

                # shuffle the order if playing all
                if opt.study == 'all':
                    # study_lvls = study_lvls
                    study_lvls = study_lvls.sample(frac=1)

                # play all of the levels
                for index, lvl_config in study_lvls.iterrows():
                    # check study type:
                    if correct_study_type(opt.study, lvl_config["lvl_type"]):
                        agent_save_path = os.path.join(
                            LSI_STEAK_STUDY_AGENT_DIR, "{lvl_type}_{lvl_vision}_{lvl_robot_aware}_{lvl_search_depth}_{lvl_kb_depth}.pkl".format(lvl_type=lvl_config["lvl_type"], lvl_vision=lvl_config["vision_bound"], lvl_robot_aware=lvl_config["vision_limit_aware"], lvl_search_depth=lvl_config["search_depth"], lvl_kb_depth=lvl_config["kb_search_depth"]))
                        value_kb_save_path = os.path.join(
                            LSI_STEAK_STUDY_AGENT_DIR,
                            "{lvl_type}_{lvl_vision}_{lvl_robot_aware}_{lvl_search_depth}_{lvl_kb_depth}_v_and_kb.pkl".format(lvl_type=lvl_config["lvl_type"], lvl_vision=lvl_config["vision_bound"], lvl_robot_aware=lvl_config["vision_limit_aware"], lvl_search_depth=lvl_config["search_depth"], lvl_kb_depth=lvl_config["kb_search_depth"]))
                        print(lvl_config["lvl_type"])
                        if not opt.human_play:
                            results = agents_play(lvl_config["lvl_str"],
                                                agent_save_path=agent_save_path,
                                                value_kb_save_path = value_kb_save_path,
                                                VISION_LIMIT = lvl_config["vision_limit"],
                                                VISION_BOUND = int(lvl_config["vision_bound"]),
                                                VISION_LIMIT_AWARE = lvl_config["vision_limit_aware"],
                                                EXPLORE =  lvl_config["explore"],
                                                agent_unstuck= lvl_config["agent_unstuck"],
                                                human_unstuck= lvl_config["human_unstuck"],
                                                SEARCH_DEPTH= lvl_config["search_depth"],
                                                KB_SEARCH_DEPTH= lvl_config["kb_search_depth"])
                        else:
                            overcooked_game = human_play_obj(lvl_config["lvl_str"],
                                         agent_save_path=agent_save_path,
                                        value_kb_save_path=value_kb_save_path)
                            dispatcher = UdpToPygame()
                            while True:
                                dispatcher.update()
                                for event in pygame.event.get():
                                    if event.type == pygame.USEREVENT:
                                        print('recieved event from port')
                                        data = event.data.decode()
                                        state_dict = json.loads(data)
                                        # env_obj = map_dict_to_objects(state)
                                        state_obj = map_dict_to_state(state_dict)
                                        
                                        # compare pygame
                                        robot_action = overcooked_game.agent.action(state_obj)[0]

                                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

                                        #while True:
                                        sock.sendto(json.dumps(robot_action).encode(), ("127.0.0.1", 15007))

                            # overcooked_game.on_execute(dispatcher)
                        # write the results
                        # if lvl_config["lvl_type"] != "trial":
                        #     write_to_human_exp_log(human_log_csv, results,
                        #                            lvl_config)

        # loading an existing study and continue running it.
        else:
            log_index = opt.log_index
            assert int(log_index) >= 0
            human_log_csv, human_log_data = load_human_log_data(log_index)

            # find levels need to run and play them
            for lvl_type in DETAILED_STUDY_TYPES:
                if lvl_type not in human_log_data["lvl_type"].to_list():
                    lvl_config = study_lvls[study_lvls["lvl_type"] ==
                                            lvl_type].iloc[0]
                    lvl_str = lvl_config["lvl_str"]
                    print(lvl_config["lvl_type"])
                    print(lvl_str)
                    agent_save_path = os.path.join(
                        LSI_STEAK_STUDY_AGENT_DIR,
                        "{lvl_type}_{lvl_vision}_{lvl_robot_aware}_{lvl_search_depth}_{lvl_kb_depth}.pkl".format(lvl_type=lvl_config["lvl_type"], lvl_vision=lvl_config["vision_bound"], lvl_robot_aware=lvl_config["vision_limit_aware"], lvl_search_depth=lvl_config["search_depth"], lvl_kb_depth=lvl_config["kb_search_depth"]))
                    value_kb_save_path = os.path.join(
                        LSI_STEAK_STUDY_AGENT_DIR,
                        "{lvl_type}_{lvl_vision}_{lvl_robot_aware}_{lvl_search_depth}_{lvl_kb_depth}_v_and_kb.pkl".format(lvl_type=lvl_config["lvl_type"], lvl_vision=lvl_config["vision_bound"], lvl_robot_aware=lvl_config["vision_limit_aware"], lvl_search_depth=lvl_config["search_depth"], lvl_kb_depth=lvl_config["kb_search_depth"]))
                    if not opt.human_play:
                        results = agents_play(lvl_str,
                                        agent_save_path=agent_save_path,
                                        value_kb_save_path=value_kb_save_path,
                                        VISION_LIMIT = lvl_config["vision_limit"],
                                        VISION_BOUND = int(lvl_config["vision_bound"]),
                                        VISION_LIMIT_AWARE = lvl_config["vision_limit_aware"],
                                        EXPLORE = lvl_config["explore"],
                                        agent_unstuck = lvl_config["agent_unstuck"],
                                        human_unstuck = lvl_config["human_unstuck"],
                                        SEARCH_DEPTH= lvl_config["search_depth"],
                                        KB_SEARCH_DEPTH= lvl_config["kb_search_depth"])
                    else:
                        results = human_play(lvl_str,
                                        agent_save_path=agent_save_path,
                                        value_kb_save_path=value_kb_save_path)
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

                # replay the game
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

            # replay the game
            replay_with_joint_actions(lvl_str, joint_actions, log_dir=os.path.join(LSI_STEAK_STUDY_RESULT_DIR, log_index), log_name=lvl_type, view_angle=vision_bound)

            if NO_FOG_COPY and vision_bound > 0:
                replay_with_joint_actions(lvl_str, joint_actions, log_dir=os.path.join(LSI_STEAK_STUDY_RESULT_DIR, log_index), log_name=lvl_type+'_nofog', view_angle=0)


class Player:
    def __init__(self):
        self.active_log = []
        self.held_object = None
        self.num_ingre_held = 0
        self.num_plate_held = 0
        self.num_served = 0
        self.orientation = []
        self.pos_and_or = []
        self.position = []
        self.stuck_log = []

class Env:
    def __init__(self):
        self.all_objects_list = []
        self.curr_order = 'steak'
        self.next_order = 'steak'
        self.num_orders_remaining = 2
        self.obj_count = 0
        self.objects = {}
        self.order_list = ['steak', 'steak']
        self.player_objects_by_type = {}
        self.player_orientations = [[0, -1], [0, -1]]
        self.player_positions = [[6, 7], [10, 6]]
        self.players = [Player(), Player()]
        self.players_pos_and_or = (
            ((6, 7), (0, -1)),
            ((10, 6), (0, -1))
        )

def map_dict_to_objects(data_dict):
    game_data = Env()

    # Map the dictionary data to the GameData object
    game_data.all_objects_list = data_dict.get('all_objects_list', [])
    game_data.curr_order = data_dict.get('curr_order', 'steak')
    game_data.next_order = data_dict.get('next_order', 'steak')
    game_data.num_orders_remaining = data_dict.get('num_orders_remaining', 2)
    game_data.obj_count = data_dict.get('obj_count', 0)
    game_data.objects = data_dict.get('objects', {})
    game_data.order_list = data_dict.get('order_list', [])
    game_data.player_objects_by_type = data_dict.get('player_objects_by_type', {})
    game_data.player_orientations = tuple(map(tuple,data_dict.get('player_orientations', ())))

    game_data.player_positions = tuple(map(tuple,data_dict.get('player_positions', ())))
    game_data.players_pos_and_or = data_dict.get('players_pos_and_or', [])
    
    # Map the player data to the Player objects
    for i, player_data in enumerate(data_dict.get('players', [])):
        player = game_data.players[i]
        player.active_log = player_data.get('active_log', [])
        player.held_object = player_data.get('held_object', None)
        player.num_ingre_held = player_data.get('num_ingre_held', 0)
        player.num_plate_held = player_data.get('num_plate_held', 0)
        player.num_served = player_data.get('num_served', 0)
        player.orientation = tuple(player_data.get('orientation', ()))
        player.pos_and_or = tuple(map(tuple,player_data.get('pos_and_or', ())))
        player.position = tuple(player_data.get('position', ()))
        player.stuck_log = player_data.get('stuck_log', [])

    return game_data

def map_dict_to_state(dict):
    ppo_list = dict.get('players_pos_and_or', [])
    players_pos_and_or = ((tuple(ppo_list[0][0]),tuple(ppo_list[0][1])),(tuple(ppo_list[1][0]),tuple(ppo_list[1][1])))

    obj = OvercookedState([
        PlayerState(*player_pos_and_or)
        for player_pos_and_or in players_pos_and_or
    ],
    objects=dict.get('objects', {}),
    order_list = dict.get('order_list', []))

    return obj


main()