import pygame
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np
import gc
import time

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, SteakHouseGridworld, OvercookedState, Direction, Action, PlayerState, ObjectState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import overcooked_ai_py.agents.agent as agent
import overcooked_ai_py.planning.planners as planners
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai_py.utils import load_dict_from_file

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
    def __init__(self, env, agent, agent2, player_idx, slow_time):
        self._running = True
        self._display_surf = None
        self.env = env
        self.agent = agent
        self.agent2 = agent2
        self.agent_idx = player_idx
        self.slow_time = slow_time
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
            # print("TOT rew", self.env.cumulative_sparse_rewards)
            self._running = False


    def step_env(self, my_action):
        agent_action = self.agent.action(self.env.state)[0]
        agent2_action = my_action #self.agent2.action(self.env.state)[0]

        if self.agent_idx == 0:
            joint_action = (agent_action, agent2_action)
        else:
            joint_action = (agent2_action, agent_action)

        next_state, timestep_sparse_reward, done, info = self.env.step(joint_action)

        # print(self.env)
        self.env.render()
        # print("Curr reward: (sparse)", timestep_sparse_reward, "\t(dense)", info["shaped_r_by_agent"])
        # print(self.env.t)
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

        while( self._running ):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()

if __name__ == "__main__" :

    # np.random.seed(0)
    start_time = time.time()
    layout_name = 'steak_island2' #'steak_island2' #'steak_parrallel'  'steak_tshape'
    scenario_1_mdp = SteakHouseGridworld.from_layout_name(layout_name,  num_items_for_steak=1, chop_time=2, wash_time=2, start_order_list=['steak', 'steak'], cook_time=10)
    # start_state = OvercookedState(
    #     [P((2, 1), s, Obj('onion', (2, 1))),
    #      P((3, 2), s)],
    #     {}, order_list=['onion','onion'])
    env = OvercookedEnv.from_mdp(scenario_1_mdp, horizon = 200)

    COUNTERS_PARAMS = {
        'start_orientations': True,
        'wait_allowed': True,
        'counter_goals': [],
        'counter_drop': scenario_1_mdp.terrain_pos_dict['X'],
        'counter_pickup': scenario_1_mdp.terrain_pos_dict['X'],
        'same_motion_goals': True
    }

    # ml_action_manager = planners.MediumLevelActionManager(scenario_1_mdp, NO_COUNTERS_PARAMS)
    # hmlp = planners.HumanMediumLevelPlanner(scenario_1_mdp, ml_action_manager, [0.5, (1.0-0.5)], 0.5)
    # human_agent = agent.biasHumanModel(ml_action_manager, [0.5, (1.0-0.5)], 0.5, auto_unstuck=True)
    VISION_LIMIT = True
    VISION_BOUND = 60
    VISION_LIMIT_AWARE = False
    EXPLORE = False
    SEARCH_DEPTH = 5
    KB_SEARCH_DEPTH = 3
    mlp = planners.MediumLevelPlanner.from_pickle_or_compute(scenario_1_mdp, COUNTERS_PARAMS, force_compute=True)  
    human_agent = agent.SteakLimitVisionHumanModel(mlp, env.state, auto_unstuck=True, explore=EXPLORE, vision_limit=VISION_LIMIT, vision_bound=VISION_BOUND, debug=True)
    human_agent.set_agent_index(1)

    # human_agent = agent.GreedySteakHumanModel(mlp)
    # human_agent = agent.CoupledPlanningAgent(mlp)

    qmdp_start_time = time.time()
    # mdp_planner = planners.SteakHumanSubtaskQMDPPlanner.from_pickle_or_compute(scenario_1_mdp, COUNTERS_PARAMS, force_compute_all=True, jmp = mlp.ml_action_manager.joint_motion_planner, vision_limited_human=human_agent)
    mdp_planner = None
    
    if not VISION_LIMIT_AWARE and VISION_LIMIT:
        non_limited_human = agent.SteakLimitVisionHumanModel(mlp, env.state, auto_unstuck=True, explore=EXPLORE, vision_limit=False, vision_bound=0, debug=True)
        non_limited_human.set_agent_index(1)
        mdp_planner = planners.SteakKnowledgeBasePlanner.from_pickle_or_compute(scenario_1_mdp, COUNTERS_PARAMS, force_compute_all=True, jmp = mlp.ml_action_manager.joint_motion_planner, vision_limited_human=non_limited_human, debug=True, search_depth=SEARCH_DEPTH, kb_search_depth=KB_SEARCH_DEPTH)
    else:
        limited_human = agent.SteakLimitVisionHumanModel(mlp, env.state, auto_unstuck=True, explore=EXPLORE, vision_limit=VISION_LIMIT, vision_bound=VISION_BOUND, debug=True)
        limited_human.set_agent_index(1)
        mdp_planner = planners.SteakKnowledgeBasePlanner.from_pickle_or_compute(scenario_1_mdp, COUNTERS_PARAMS, force_compute_all=True, jmp = mlp.ml_action_manager.joint_motion_planner, vision_limited_human=limited_human, debug=True, search_depth=SEARCH_DEPTH, kb_search_depth=KB_SEARCH_DEPTH)

    ai_agent = agent.MediumQMdpPlanningAgent(mdp_planner, greedy=True, auto_unstuck=True, low_level_action_flag=True, vision_limit=VISION_LIMIT)
    # ai_agent = agent.QMDPAgent(mlp, env)
    # ai_agent = agent.GreedySteakHumanModel(mlp)

    ai_agent.set_agent_index(0)

    del mlp, mdp_planner
    gc.collect()

    agent_pair = agent.AgentPair(ai_agent, human_agent) # if use QMDP, the first agent has to be the AI agent
    print("It took {} seconds for planning".format(time.time() - start_time))
    total_t = 0
    for i in range(1):
        game_start_time = time.time()
        s_t, joint_a_t, r_t, done_t = env.run_agents(agent_pair, include_final_state=True, display=True)
        # print("It took {} seconds for qmdp to compute".format(game_start_time - qmdp_start_time))
        # print("It took {} seconds for playing the entire level".format(time.time() - game_start_time))
        # print("It took {} seconds to plan".format(time.time() - start_time))
        env = OvercookedEnv.from_mdp(scenario_1_mdp, horizon = 100)
        done = False
        total_t += len(s_t)
    print('Total timesteps =', total_t)
    t = 0
    scenario_1_mdp = SteakHouseGridworld.from_layout_name(layout_name,  num_items_for_steak=1, chop_time=2, wash_time=2, start_order_list=['steak', 'steak'], cook_time=10)
    env = OvercookedEnv.from_mdp(scenario_1_mdp, horizon = 200)
    while not done:
        if t >= 0 and t <= len(s_t):
            if VISION_LIMIT: 
                env.render("fog", view_angle=VISION_BOUND)
            else:
                env.render()
            time.sleep(0.1)
        agent1_action = s_t[t][1][0]
        agent2_action = s_t[t][1][1]
        joint_action = (tuple(agent1_action) if isinstance(
            agent1_action, list) else agent1_action, tuple(agent2_action)
                        if isinstance(agent2_action, list) else agent2_action)
        next_state, timestep_sparse_reward, done, info = env.step(joint_action)
        t += 1
        time.sleep(0.4)

    del scenario_1_mdp, env, agent_pair, ai_agent, human_agent
    gc.collect()

    # theApp = App(env, ai_agent, human_agent, player_idx=0, slow_time=False)
    # theApp.on_execute()
    # print("It took {} seconds for playing the entire level".format(time.time() - start_time))
