import pygame
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np
import gc
import time

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, Direction, Action, PlayerState, ObjectState
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
        agent2_action = self.agent2.action(self.env.state)[0]

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


COUNTERS_PARAMS = {
    'start_orientations': True,
    'wait_allowed': True,
    'counter_goals': [],
    'counter_drop': [(0, 1), (6, 2), (2, 4), (0, 4), (3, 4), (0, 0), (3, 1), (6, 1), (0, 3), (6, 4), (3, 0), (0, 2), (5, 0), (6, 0), (1, 0), (3, 2), (6, 3)],
    'counter_pickup': [(0, 1), (6, 2), (2, 4), (0, 4), (3, 4), (0, 0), (3, 1), (6, 1), (0, 3), (6, 4), (3, 0), (0, 2), (5, 0), (6, 0), (1, 0), (3, 2), (6, 3)],
    'same_motion_goals': True
}

if __name__ == "__main__" :

    # np.random.seed(0)
    start_time = time.time()
    scenario_1_mdp = OvercookedGridworld.from_layout_name('10x15_test1', start_order_list=['onion','onion'], num_items_for_soup=2, cook_time=10)
    # start_state = OvercookedState(
    #     [P((2, 1), s, Obj('onion', (2, 1))),
    #      P((3, 2), s)],
    #     {}, order_list=['onion','onion'])
    env = OvercookedEnv.from_mdp(scenario_1_mdp, horizon = 100)

    # ml_action_manager = planners.MediumLevelActionManager(scenario_1_mdp, NO_COUNTERS_PARAMS)

    # hmlp = planners.HumanMediumLevelPlanner(scenario_1_mdp, ml_action_manager, [0.5, (1.0-0.5)], 0.5)
    # human_agent = agent.biasHumanModel(ml_action_manager, [0.5, (1.0-0.5)], 0.5, auto_unstuck=True)

    mlp = planners.MediumLevelPlanner.from_pickle_or_compute(scenario_1_mdp, NO_COUNTERS_PARAMS, force_compute=True)  
    human_agent = agent.GreedyHumanModel(mlp)
    # human_agent = agent.CoupledPlanningAgent(mlp)

    qmdp_start_time = time.time()
    mdp_planner = planners.HumanSubtaskQMDPPlanner.from_pickle_or_compute(scenario_1_mdp, NO_COUNTERS_PARAMS, force_compute_all=True)
    ai_agent = agent.MediumQMdpPlanningAgent(mdp_planner, greedy=True, auto_unstuck=True)
    # ai_agent = agent.QMDPAgent(mlp, env)

    ai_agent.set_agent_index(0)
    human_agent.set_agent_index(1)

    del mlp, mdp_planner
    gc.collect()

    agent_pair = agent.AgentPair(ai_agent, human_agent) # if use QMDP, the first agent has to be the AI agent
    print("It took {} seconds for planning".format(time.time() - start_time))
    total_t = 0
    for i in range(100):
        game_start_time = time.time()
        s_t, joint_a_t, r_t, done_t = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)
        # print("It took {} seconds for qmdp to compute".format(game_start_time - qmdp_start_time))
        # print("It took {} seconds for playing the entire level".format(time.time() - game_start_time))
        # print("It took {} seconds to plan".format(time.time() - start_time))
        env = OvercookedEnv.from_mdp(scenario_1_mdp, horizon = 100)
        done = False
        total_t += len(s_t)
    print('Average timesteps =', total_t/10.0)
    t = 0
    scenario_1_mdp = OvercookedGridworld.from_layout_name('10x15_test1', start_order_list=['onion','onion'], num_items_for_soup=2, cook_time=10)
    env = OvercookedEnv.from_mdp(scenario_1_mdp, horizon = 100)
    while not done:
        if t >= 0 and t <= len(s_t):
            # env.render("blur", time_left=t)
            env.render()
            time.sleep(0.1)
        agent1_action = s_t[t][1][0]
        agent2_action = s_t[t][1][1]
        joint_action = (tuple(agent1_action) if isinstance(
            agent1_action, list) else agent1_action, tuple(agent2_action)
                        if isinstance(agent2_action, list) else agent2_action)
        next_state, timestep_sparse_reward, done, info = env.step(joint_action)
        t += 1
        tmp = input()

    del scenario_1_mdp, env, agent_pair, ai_agent, human_agent
    gc.collect()

    # theApp = App(env, ai_agent, human_agent, player_idx=0, slow_time=False)
    # theApp.on_execute()
    # print("It took {} seconds for playing the entire level".format(time.time() - start_time))
