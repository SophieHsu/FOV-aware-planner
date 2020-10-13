import pygame
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np

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

valid_counters = [(5, 3)]
one_counter_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': valid_counters,
    'counter_drop': valid_counters,
    'counter_pickup': [],
    'same_motion_goals': True
}

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState
DISPLAY = True

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

    scenario_1_mdp = OvercookedGridworld.from_layout_name('five_by_five', start_order_list=['any','any'], cook_time=2)
    # start_state = OvercookedState(
    #     [P((2, 1), s, Obj('onion', (2, 1))),
    #      P((3, 2), s)],
    #     {}, order_list=['onion','onion'])
    env = OvercookedEnv.from_mdp(scenario_1_mdp)

    mlp = planners.MediumLevelPlanner.from_pickle_or_compute(scenario_1_mdp, NO_COUNTERS_PARAMS, force_compute=True)

    # a0 = agent.GreedyHumanModel(mlp)


    hmlp = planners.HumanMediumLevelPlanner(scenario_1_mdp, mlp, one_goal=0)
    a0 = agent.oneGoalHumanModel(mlp, 'Onion cooker', auto_unstuck=True)


    mdp_planner = planners.HumanAwareMediumMDPPlanner.from_pickle_or_compute(scenario_1_mdp, NO_COUNTERS_PARAMS, hmlp, mlp, force_compute_all=True)
    a1 = agent.MediumMdpPlanningAgent(mdp_planner, env)

    # # a1 = agent.oneGoalHumanModel(mlp, 'Soup server', auto_unstuck=True)
    # a1 = agent.biasHumanModel(mlp, [0.3, 0.7], 0.3, auto_unstuck=True)

    # mdp_planner = MdpPlanner.from_pickle_or_compute(scenario_1_mdp, a0, 0, NO_COUNTERS_PARAMS, force_compute_all = False, force_compute_more=False)#, custom_filename='gen1_basic_1-3_mdp_70.pkl')
    # a1 = MdpPlanningAgent(a0, mdp_planner, env)

    agent_pair = agent.AgentPair(a0, a1)

    s_t, joint_a_t, r_t, done_t = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)

    # print(s_t, joint_a_t, r_t, done_t)

    # theApp = App(env, a0, a1, player_idx=0, slow_time=False)
    # theApp.on_execute()