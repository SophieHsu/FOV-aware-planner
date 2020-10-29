import os
import json
import time
import argparse
import pygame
from overcooked_ai_pcg.helper import lvl_str2grid, CONFIG
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

def render_blur(joint_action_log, log_dir, lb, ub):
    grid = lvl_str2grid(joint_action_log["lvl_str"])
    joint_actions = joint_action_log["joint_actions"]

    # need to set up and run game mannully because
    mdp = OvercookedGridworld.from_grid(grid, CONFIG)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=100)

    done = False
    t = 0
    while not done:
        if t >= lb and t <= ub:
            env.render(mode="blur")
            time.sleep(0.1)
        agent1_action = joint_actions[t][0]
        agent2_action = joint_actions[t][1]
        joint_action = (tuple(agent1_action) if isinstance(
            agent1_action, list) else agent1_action, tuple(agent2_action)
                        if isinstance(agent2_action, list) else agent2_action)
        next_state, timestep_sparse_reward, done, info = env.step(joint_action)
        t += 1

    # save the rendered blur image
    pygame.image.save(
        env.mdp.viewer,
        os.path.join(log_dir, "blurred_play_%sto%s.png" % (str(lb), str(ub))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--log_file',
                        help='path of joint action log file',
                        required=True)
    parser.add_argument('-lb',
                        '--lower_bound',
                        help='lower bound of the timestep to render',
                        required=True)
    parser.add_argument('-ub',
                        '--upper_bound',
                        help='upper bound of the timestep to render',
                        required=True)
    opt = parser.parse_args()
    with open(opt.log_file, 'r') as f:
        joint_action_log = json.load(f)
    log_dir = os.path.split(opt.log_file)[0]

    # get upper and lower bounds
    lb = int(opt.lower_bound)
    ub = int(opt.upper_bound)
    assert (lb <= ub)
    render_blur(joint_action_log, log_dir, lb, ub)