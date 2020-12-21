import argparse
import json
import os
import subprocess
import time
import pygame
import numpy as np


from overcooked_ai_py.agents.agent import *

from overcooked_ai_pcg import LSI_CONFIG_EXP_DIR
from overcooked_ai_pcg.helper import run_overcooked_game
from overcooked_ai_pcg.helper import read_in_lsi_config
from overcooked_ai_pcg.LSI.qd_algorithms import Individual

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner, MediumLevelActionManager, HumanSubtaskQMDPPlanner


from overcooked_ai_py.planning.planners import Heuristic


class DocplexFailedError(Exception):
    pass


NO_COUNTERS_PARAMS = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}



def main(config, option):
#    _, _, _, agent_configs = read_in_lsi_config(config)

    #if option == "full":
    #    lvl_str = """XXPXX
    #                 X2  O
    #                 X1  X
    #                 XXDSX
    #              """
    # lvl_str = """XXXPPXXX
    #              X  2   X
    #              D XXXX S
    #              X  1   X
    #              XXXOOXXX
    #              """
    #  lvl_str = """XXXXXXXXXXXXXXX
    #               O 1     XX    D
    #               X  X2XXXXXXXX S
    #               O XX     XXXX X
    #               X         X   X
    #               O          X  X
    #               X  XXXXXXX    X
    #               X  XXXX  PXXX X
    #               X          X  X
    #               XXXXXXXXXXXXXXX
    #             """



    scenario_1_mdp = OvercookedGridworld.from_layout_name('sn_level', start_order_list=['onion','onion'], num_items_for_soup=1, cook_time=5)
    env = OvercookedEnv.from_mdp(scenario_1_mdp, horizon = 1000)
    #opt.config = "LSI/data/config/experiment/CMAME_workloads_diff_QMDP.tml"  
    #_, _, elite_map_config, agent_configs = read_in_lsi_config(opt.config)

    #mlp_planner = MediumLevelPlanner(scenario_1_mdp, NO_COUNTERS_PARAMS)
    #joint_plan = \
    #        mlp_planner.get_ml_plan(
    #            env.state,
    #            Heuristic(mlp_planner.mp).simple_heuristic)

    mlp_planner = MediumLevelPlanner(scenario_1_mdp, NO_COUNTERS_PARAMS)
#    print("worker(%d): Planning..." % (worker_id))
    # joint_plan = \
    #     mlp_planner.get_low_level_action_plan(
    #         env.state,
    #         Heuristic(mlp_planner.mp).simple_heuristic,
    #         delivery_horizon=2,
    #         goal_info=True)

    # plan1 = []
    # plan2 = []
    # for joint_action in joint_plan:
    #    action1, action2 = joint_action
    #    plan1.append(action1)
    #    plan2.append(action2)

    #agent1 = FixedPlanAgent(plan1)
    #agent2 = FixedPlanAgent(plan2)
    mdp_planner = HumanSubtaskQMDPPlanner.from_pickle_or_compute(scenario_1_mdp, NO_COUNTERS_PARAMS, mlp_planner.ml_action_manager, mlp=mlp_planner, force_compute_all=True)
    agent1 = MediumQMdpPlanningAgent(mdp_planner, auto_unstuck=True)
    # agent1 = QMDPAgent(mlp_planner, env)
    agent2 = CoupledPlanningAgent(mlp_planner)

    # agent2 = RandomAgent(mlp_planner)


    agent1.set_agent_index(0)
    agent2.set_agent_index(1)


    done = False
    total_sparse_reward = 0
    last_state = None
    timestep = 0
    np.random.seed(1)

    # Saves when each soup (order) was delivered
    checkpoints = [env.horizon - 1] * env.num_orders
    cur_order = 0

    start_time = time.time()
    # store all actions
    joint_actions = []
    while not done:
        env.render()
        time.sleep(1)
        joint_action = (agent1.action(env.state)[0],
                        agent2.action(env.state)[0])
        # print(joint_action)
        joint_actions.append(joint_action)

        next_state, timestep_sparse_reward, done, info = env.step(joint_action)
        total_sparse_reward += timestep_sparse_reward
        pot_states_dict = scenario_1_mdp.get_pot_states(next_state)
        #print(pot_states_dict['onion'])
        #from IPython import embed
        #embed()

        if timestep_sparse_reward > 0:
            checkpoints[cur_order] = timestep
            cur_order += 1

        last_state = next_state
        timestep += 1

    elapsed_time = time.time() - start_time
    print("Elapsed time: " + str(elapsed_time))

#    ind = Individual()
    #ind.level = lvl_str

#    fitness, _, _, _, _, _, _ = run_overcooked_game(ind, agent_configs[0], render=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        help='path of experiment config file',
                        required=False,
                        default=os.path.join(
                            LSI_CONFIG_EXP_DIR,
                            "MAPELITES_workloads_diff_fixed_plan.tml"))
    parser.add_argument('-o',
                        '--option',
                        help="option of pcg pipeline. Can be one of 'full' and milp_only'.",
                        default="full")
    opt = parser.parse_args()
    main(opt.config, option=opt.option)

