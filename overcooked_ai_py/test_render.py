import time
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import *

mdp = OvercookedGridworld.from_layout_name("simple_tomato")
env = OvercookedEnv.from_mdp(mdp)
agent1 = RandomAgent(all_actions=True)
agent2 = RandomAgent(all_actions=True)
agent1.set_agent_index(0)
agent2.set_agent_index(1)
agent1.set_mdp(mdp)
agent2.set_mdp(mdp)

done = False
while not done:
    env.render()
    joint_action = (agent1.action(env.state)[0], agent2.action(env.state)[0])
    next_state, timestep_sparse_reward, done, info = env.step(joint_action)

time.sleep(60)