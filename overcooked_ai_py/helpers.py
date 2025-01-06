import csv, os
from . import *
from overcooked_ai_py.agents.agent import *
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld
from overcooked_ai_py.planning.planners import (SteakKnowledgeBasePlanner,
                                                MediumLevelPlanner)

# original overcooked_ai_pcg.helper code

BASE_PARAMS = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}


STEAK_CONFIG = {
    "start_order_list": ['steak', 'steak'],
    "cook_time": 10,
    "delivery_reward": 20,
    'num_items_for_steak': 1,
    'chop_time': 2,
    'wash_time': 2,
    "rew_shaping_params": None
}


def lvl_str2grid(lvl_str):
    """
    Turns pure string formatted lvl to grid format compatible with overcooked-AI env
    """
    return [layout_row.strip() for layout_row in lvl_str.split("\n")]#[:-1]


def init_steak_env(lvl_str, horizon=200):
    grid = lvl_str2grid(lvl_str)
    mdp = SteakHouseGridworld.from_grid(grid, STEAK_CONFIG)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=horizon)
    return env


def init_steak_qmdp_agent(env, search_depth=5, kb_search_depth=2, kb_update_delay=2, vision_limit=True, vision_bound=120, vision_limit_aware=True):
    mlp = MediumLevelPlanner.from_pickle_or_compute(env.mdp, BASE_PARAMS, force_compute=True)
    print('loading qmdp:', search_depth, kb_search_depth, vision_bound, vision_limit, kb_update_delay, vision_limit_aware)

    if vision_limit_aware:
        human_agent = SteakLimitVisionHumanModel(mlp, env.state, auto_unstuck=True, explore=False, vision_limit=vision_limit, vision_bound=vision_bound, kb_update_delay=kb_update_delay, debug=False)
    else:
        human_agent = SteakLimitVisionHumanModel(mlp, env.state, auto_unstuck=True, explore=False, vision_limit=False, vision_bound=0, kb_update_delay=0, debug=False)

    human_agent.set_agent_index(1)

    qmdp_planner = SteakKnowledgeBasePlanner.from_pickle_or_compute(
        env.mdp, BASE_PARAMS, force_compute_all=True, jmp = mlp.ml_action_manager.joint_motion_planner, vision_limited_human=human_agent, debug=False, search_depth=search_depth, kb_search_depth=kb_search_depth)

    agent = MediumQMdpPlanningAgent(qmdp_planner,
                                    greedy=False,
                                    auto_unstuck=True)
    return agent

# original overcooked_ai_pcg.LSI.human_study.py code

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



def write_row(csv_file, to_add):
    """Append a row to csv file"""
    with open(csv_file, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(to_add)

