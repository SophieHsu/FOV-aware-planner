import os
import time
import torch
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, OvercookedV1
from overcooked_ai_py.agents.agent import *
from overcooked_ai_rl.featurize_fn import *
from overcooked_ai_rl.value_functions import GaussianCNNMLPValueFunction
from overcooked_ai_rl.policies import CategoricalCNNPolicy

from garage import wrap_experiment
# from garage.torch.policies import CategoricalCNNPolicy
from garage.torch.algos import PPO
from garage.torch.optimizers import OptimizerWrapper
from garage.envs import GymEnv


def render_level(layout_name):
    _, _, env = setup_env(layout_name)
    print("size: ", (env.mdp.width, env.mdp.height))
    env.render()
    time.sleep(60)

# Green hat is agent 1/robot; Blue hat is agent 2/human
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

env = GymEnv(overcooked_v1)

policy = CategoricalCNNPolicy(
    env.spec,
    image_format="NHWC",
    kernel_sizes=(5, 3, 3),
    hidden_channels=(25, 25, 25),
)

print(policy)

vf = GaussianCNNMLPValueFunction(
    env.spec,
    image_format="NHWC",
    kernel_sizes=(5, 3, 3),
    hidden_channels=(25, 25, 25),
)

print(vf)

# done = False
# while not done:
#     env.render()
#     joint_action = (agent1.action(env.state)[0], agent2.action(env.state)[0])
#     next_state, timestep_sparse_reward, done, info = env.step(joint_action)
#     time.sleep(0.1)



done = False
next_obs = overcooked_v1.reset()
while not done:
    env.render(mode="human")
    ai_action, _ = policy.get_action(torch.Tensor(next_obs))
    next_obs, reward, done, info = overcooked_v1.step(ai_action)
    time.sleep(0.1)
    print(vf.forward(torch.Tensor(next_obs)))
    print(vf.compute_loss(torch.Tensor(next_obs), torch.Tensor([reward])))