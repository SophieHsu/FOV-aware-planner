import os, shutil
import json
import time
import argparse
import pygame
from overcooked_ai_py.helpers import lvl_str2grid, CONFIG
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

def render_blur(joint_action_log, log_dir, log_name, lb, ub):
    grid = lvl_str2grid(joint_action_log["lvl_str"])
    joint_actions = joint_action_log["joint_actions"][0]

    # need to set up and run game mannully because
    mdp = OvercookedGridworld.from_grid(grid, CONFIG)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=100)

    done = False
    t = 0
    ub = len(joint_actions) if ub > len(joint_actions) else ub

    img_dir = os.path.join(log_dir, log_name)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_name = lambda timestep: f"{img_dir}/{t:05d}.png"

    while not done:
        if t >= lb and t <= ub:
            # env.render("blur", time_step_left=ub-t, time_passed=t)
            env.render()
            time.sleep(0.1)

            if img_name is not None:
                cur_name = img_name(t)
                pygame.image.save(env.mdp.viewer, cur_name)

        agent1_action = joint_actions[t][0]
        agent2_action = joint_actions[t][1]
        joint_action = (tuple(agent1_action) if isinstance(
            agent1_action, list) else agent1_action, tuple(agent2_action)
                        if isinstance(agent2_action, list) else agent2_action)
        next_state, timestep_sparse_reward, done, info = env.step(joint_action)
        t += 1
        # print(t)
        # tmp = input()

    os.system("ffmpeg -r 5 -i \"{}%*.png\"  {}{}.mp4".format(img_dir+'/', log_dir+'/', log_name))
    shutil.rmtree(img_dir) 


    # save the rendered blur image
    pygame.image.save(
        env.mdp.viewer,
        os.path.join(log_dir, log_name+"_blurred_%s_to_%s.png" % (str(lb), str(ub))))


    # print("fitness =", joint_action_log["fitness"])
    # print("score =", joint_action_log["score"])
    # print("checkpoints =", joint_action_log["checkpoints"])
    # print("player_workloads =", joint_action_log["player_workloads"])
    # print("concurr_active =", joint_action_log["concurr_active"])
    # print("stuck_time =", joint_action_log["stuck_time"])
    # print("rand_seed =", joint_action_log["rand_seed"])

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

    log_dir, log_json_file = os.path.split(opt.log_file)
    log_name = log_json_file.split('.')[0]
    print(opt.upper_bound)
    # get upper and lower bounds
    lb = int(opt.lower_bound)
    ub = int(opt.upper_bound)
    assert (lb <= ub)
    render_blur(joint_action_log, log_dir, log_name, lb, ub)