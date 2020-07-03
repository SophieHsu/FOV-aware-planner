import time
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

mdp = OvercookedGridworld.from_layout_name("corridor")
env = OvercookedEnv.from_mdp(mdp)
env.render()
time.sleep(60)