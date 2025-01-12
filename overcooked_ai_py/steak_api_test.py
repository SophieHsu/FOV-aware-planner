import argparse
import json
import os
import pickle
import pprint
import socket
import pygame
import time

from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld, OvercookedState, Direction, Action, PlayerState, ObjectState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import overcooked_ai_py.agents.agent as agent
import overcooked_ai_py.planning.planners as planners
from overcooked_ai_py.helpers import init_steak_qmdp_agent, create_human_exp_log

NO_COUNTERS_PARAMS = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState
DISPLAY = False

class App:
    """Class to run an Overcooked Gridworld game, leaving one of the players as fixed.
    Useful for debugging. Most of the code from http://pygametutorials.wikidot.com/tutorials-basic."""
    def __init__(self, env, agent, agent2, player_idx, slow_time, layout_name=None):
        self._running = True
        self._display_surf = None
        self.env = env
        self.agent = agent
        self.agent2 = agent2
        self.agent_idx = player_idx
        self.slow_time = slow_time
        self.layout_name = layout_name
        self.log = None

    def on_init(self):
        pygame.init()

        # Adding pre-trained agent as teammate
        self.agent.set_agent_index(self.agent_idx)
        self.agent.set_mdp(self.env.mdp)
        self.agent2.set_agent_index(self.agent_idx+1)
        self.agent2.set_mdp(self.env.mdp)

        self.env.render()
        self._running = True

    def on_event(self, event):
        done = False

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

                done = self.step_env(action)

                if self.slow_time and not done:
                    for _ in range(2):
                        action = Action.STAY
                        done = self.step_env(action)
                        if done:
                            break

        if event.type == pygame.QUIT or done:
            self._running = False


    def step_env(self, my_action):
        agent_action = self.agent.action(self.env.state)[0]
        agent2_action = my_action #self.agent2.action(self.env.state)[0]

        if self.agent_idx == 0:
            joint_action = (agent_action, agent2_action)
        else:
            joint_action = (agent2_action, agent_action)

        next_state, timestep_sparse_reward, done, info = self.env.step(joint_action)

        self.env.render()
        return done

    def on_loop(self):
        pass
    def on_render(self):
        pass

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False
        dispatcher = UdpToPygame()
        while( self._running ):
            dispatcher.update()
            for event in pygame.event.get():
                self.on_event(event)
                if event.type == pygame.USEREVENT:

                    print('recieved event from port')
                    data = event.data.decode()
                    transfer_dict = json.loads(data)
                    state_dict = transfer_dict['ovc_state']
                    ids_dict = transfer_dict['ids_dict']
                    print('ids length from igibson: ', len(ids_dict))
                    pprint.pprint(ids_dict)
                    state_obj = OvercookedState.from_dict(state_dict, obj_count=len(ids_dict))
                    self.agent.mdp_planner.igibson_overwrite_object_id_dict(ids_dict)
                    # compare pygame
                    robot_action,action_probs,q = self.agent.action(state_obj)
                    return_data = {'action': robot_action, 'q': q.tolist()}
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

                    sock.sendto(json.dumps(return_data).encode(), ("127.0.0.1", 15007))

            self.on_loop()
            self.on_render()
        value_kb_save_path = f"data/planners/{self.layout_name}_kb.pkl"
        if value_kb_save_path is not None:
                with open(value_kb_save_path, 'wb') as f:
                    pickle.dump([self.agent.mdp_planner.world_state_cost_dict, self.agent.mdp_planner.track_state_kb_map], f)
        self.on_cleanup()

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

def load_steak_qmdp_agent(env, agent_save_path, value_kb_save_path):
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
        ai_agent = init_steak_qmdp_agent(env, search_depth=SEARCH_DEPTH, kb_search_depth=KB_SEARCH_DEPTH, vision_limit=VISION_LIMIT, vision_bound=VISION_BOUND, kb_update_delay=KB_UPDATE_DELAY, vision_limit_aware=VISION_LIMIT_AWARE)
        ai_agent.set_agent_index(0)
        if agent_save_path is not None:
            with open(agent_save_path, 'wb') as f:
                pickle.dump(ai_agent, f)
    return ai_agent

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout",
        "-l",
        default=None,
        help="layout map to load",
    )

    parser.add_argument(
        "--vision",
        "-v",
        default=None,
        help="0 for unaware 1 for aware",
    )

    parser.add_argument(
        "--kbnoact",
        "-k",
        default=None,
        help="0 for unaware 1 for aware",
    )

    args = parser.parse_args()

    start_time = time.time()

    if args.layout is not None:
        layout_name = args.layout
    else:
        layout_name = 'steak_side_4'
    scenario_1_mdp = SteakHouseGridworld.from_layout_name(layout_name,  num_items_for_steak=1, chop_time=2, wash_time=2, start_order_list=['steak', 'steak', 'steak', 'steak'], cook_time=10)
    env = OvercookedEnv.from_mdp(scenario_1_mdp, horizon = 200)

    COUNTERS_PARAMS = {
        'start_orientations': True,
        'wait_allowed': True,
        'counter_goals': [],
        'counter_drop': scenario_1_mdp.terrain_pos_dict['X'],
        'counter_pickup': scenario_1_mdp.terrain_pos_dict['X'],
        'same_motion_goals': True
    }

    VISION_LIMIT = True
    VISION_BOUND = 120

    
    EXPLORE = False

    SEARCH_DEPTH = 5
    
    if args.vision == "1":
        KB_SEARCH_DEPTH = 3
    else:
        KB_SEARCH_DEPTH = 0
        
    print('kb search depth = ', KB_SEARCH_DEPTH)

    KB_UPDATE_DELAY = 3

    print(args.vision)
    if args.vision == "1":
        VISION_LIMIT_AWARE = True
        print("aware")
        ai_agent = load_steak_qmdp_agent(env, f'overcooked_ai_py/data/planners/{layout_name}_steak_knowledge_aware_qmdp.pkl', f'overcooked_ai_py/data/planners/{layout_name}_aware_kb.pkl')
    elif args.vision == "0":
        VISION_LIMIT_AWARE = False
        print('unaware')
        ai_agent = load_steak_qmdp_agent(env, f'overcooked_ai_py/data/planners/{layout_name}_steak_knowledge_unaware_qmdp.pkl', f'overcooked_ai_py/data/planners/{layout_name}_unaware_kb.pkl')
    else:
        VISION_LIMIT_AWARE = True
        print('unaware')
        ai_agent = load_steak_qmdp_agent(env, f'overcooked_ai_py/data/planners/{layout_name}_steak_knowledge_unaware_qmdp.pkl', f'overcooked_ai_py/data/planners/{layout_name}_kb.pkl')

    ai_agent.set_agent_index(0)


    
    mlp = planners.MediumLevelPlanner.from_pickle_or_compute(scenario_1_mdp, COUNTERS_PARAMS, force_compute=True)  
    human_agent = agent.SteakLimitVisionHumanModel(mlp, env.state, auto_unstuck=True, explore=EXPLORE, vision_limit=VISION_LIMIT, vision_bound=VISION_BOUND, kb_update_delay=KB_UPDATE_DELAY, debug=True)
    human_agent.set_agent_index(1)

    qmdp_start_time = time.time()
    mdp_planner = None

    human_log_csv = create_human_exp_log()

    theApp = App(env, ai_agent, human_agent, player_idx=0, slow_time=False, layout_name=layout_name)
    results = theApp.on_execute()
    lvl_config = {}
    lvl_config['layout_name'] = layout_name
    lvl_config['vision_limit'] = VISION_LIMIT
    lvl_config['vision_bound'] = VISION_BOUND
    lvl_config['vision_limit_aware'] = VISION_LIMIT_AWARE
    lvl_config['explore'] = EXPLORE
    lvl_config['search_depth'] = SEARCH_DEPTH
    lvl_config['kb_search_depth'] = KB_SEARCH_DEPTH
    lvl_config['kb_update_delay'] = KB_UPDATE_DELAY

    write_to_human_exp_log(human_log_csv, results, lvl_config)
    print("It took {} seconds for playing the entire level".format(time.time() - start_time))
