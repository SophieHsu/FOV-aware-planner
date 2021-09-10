import argparse, toml, os, time, json
import torch, pygame

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import (HRLTrainingAgent, GreedyHumanModel)
from overcooked_ai_py.planning.planners import (
    Heuristic, HumanSubtaskQMDPPlanner, MediumLevelActionManager, MediumLevelMdpPlanner, MediumLevelPlanner)
from overcooked_ai_rl.dqn import Qnet, setup_env_w_agents, encode_env, reset

def log_actions(log_dir, actions):
    full_path = os.path.join(log_dir, "joint_actions.json")
    if os.path.exists(full_path):
        print("Joint actions logged before, skipping...")
        return

    # log the joint actions if not logged before
    with open(full_path, "w") as f:
        json.dump({
            "joint_actions": actions,
        }, f)
        print("Joint actions saved")

def main(config, q, log_dir, human=None):
    # config overcooked env and human agent
    if not config['Env']['multi']:
        ai_agent, human_agent, env, mdp = setup_env_w_agents(config, human=human, q=q)
    else:
        env_list=os.listdir(config['Env']['layout_dir'])
        env_list.remove('base.layout')
        ai_agent, human_agent, env, mdp = setup_env_w_agents(config, len(env_list), env_list, human=human, q=q)
    h_state, env = reset(mdp, config)
    done = False

    total_sparse_reward = 0; timestep = 0
    # Saves when each soup (order) was delivered
    checkpoints = [env.horizon - 1] * env.num_orders
    cur_order = 0; last_state = None; joint_actions = []

    img_dir = log_dir
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_name = lambda timestep: f"{img_dir}/{timestep:05d}.png"

    while not done:
        env.render()
        time.sleep(0.5)

        if img_name is not None:
            cur_name = img_name(timestep)
            pygame.image.save(env.mdp.viewer, cur_name)
        
        joint_action = (ai_agent.action(env.state),
                        human_agent.action(env.state)[0])
        # print(joint_action)
        joint_actions.append(joint_action)
        next_state, timestep_sparse_reward, done, info = env.step(
            joint_action)
        total_sparse_reward += timestep_sparse_reward

        if timestep_sparse_reward > 0:
            checkpoints[cur_order] = timestep
            cur_order += 1

        last_state = next_state
        timestep += 1

    print("Fitness:", total_sparse_reward)
    log_actions(os.path.join(config["Experiment"]["log_dir"], config["Experiment"]["log_name"]), joint_actions)

    os.system("ffmpeg -r 5 -i \"{}%*.png\"  {}{}_video.mp4".format(img_dir+'/', img_dir+'/', human_agent.get_model_name()))
    for file in os.listdir(img_dir):
        if file.endswith('.png'):
            os.remove(os.path.join(img_dir, file)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--log_dir',
                        help='path of log dir',
                        required=True)
    parser.add_argument('-q',
                        '--qnet',
                        help='Qnet pth file name',
                        required=True)
    parser.add_argument('-a',
                        '--human_type',
                        help='Type of human agent (greedy_agent or random_agent) for testing',
                        default=None,
                        required=False)
    opt = parser.parse_args()

    with open(os.path.join(opt.log_dir, 'config.tml')) as f:
        config = toml.load(f)

    q = Qnet()
    q.load_state_dict(torch.load(os.path.join(opt.log_dir, opt.qnet)))
    q.eval()

    main(config, q, opt.log_dir, human=opt.human_type)

