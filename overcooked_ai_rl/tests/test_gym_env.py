import os
import time
import torch
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, OvercookedV1
from overcooked_ai_py.agents.agent import *
from overcooked_ai_rl.featurize_fn import *


def render_level(layout_name):
    _, _, env = setup_env(layout_name)
    print("size: ", (env.mdp.width, env.mdp.height))
    env.render()
    time.sleep(60)

# Green hat is agent 1/robot; Blue hat is agent2/human
def setup_env(layout_name):
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv.from_mdp(mdp, info_level = 0, horizon=100)
    agent1 = RandomAgent(all_actions=True)
    agent2 = StayAgent()
    agent1.set_agent_index(0)
    agent2.set_agent_index(1)
    agent1.set_mdp(mdp)
    agent2.set_mdp(mdp)
    return agent1, agent2, env


agent1, agent2, env = setup_env("train_gan_large/gen2_basic_6-6-4")

overcooked_v1 = OvercookedV1(agent1, agent2, env, lossless_state_featurize)

print(overcooked_v1.observation_space)
print(overcooked_v1.action_space)

# done = False
# while not done:
#     env.render()
#     joint_action = (agent1.action(env.state)[0], agent2.action(env.state)[0])
#     next_state, timestep_sparse_reward, done, info = env.step(joint_action)
#     time.sleep(0.1)



done = False
next_obs = overcooked_v1.reset()
while not done:
    env.render()
    ai_action = np.random.randn(6)
    next_obs, reward, done, info = overcooked_v1.step(ai_action)
    time.sleep(0.1)