import argparse
import os
import time
import torch
import toml
from pprint import pprint
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, OvercookedV1
from overcooked_ai_py.agents.agent import (RLTrainingAgent, GreedyHumanModel)

from overcooked_ai_py.planning.planners import (
    Heuristic, HumanAwareMediumMDPPlanner, HumanMediumLevelPlanner,
    HumanSubtaskQMDPPlanner, MediumLevelActionManager, MediumLevelMdpPlanner,
    MediumLevelPlanner)

from overcooked_ai_rl.policies import CategoricalCNNPolicy
from overcooked_ai_rl.value_functions import GaussianCNNMLPValueFunction
from overcooked_ai_rl.featurize_fn import *
from overcooked_ai_rl.optimizer_wrapper import OptimizerWrapper

from garage import wrap_experiment
from garage.torch.algos import PPO
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.trainer import Trainer
from garage.sampler import RaySampler
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.torch import set_gpu_mode


def setup_env_w_agents(config):
    """
    Setup environment and agents.

    NOTE: Green hat is agent 1/robot; Blue hat is agent 2/human
    """
    env_config, human_config = config["Env"], config["Human"]
    mdp = OvercookedGridworld.from_layout_name(
        env_config["layout_name"], **env_config["params_to_overwrite"])
    env = OvercookedEnv.from_mdp(mdp,
                                 info_level=0,
                                 horizon=env_config["horizon"])
    ai_agent = RLTrainingAgent()
    ai_agent.set_agent_index(0)
    ai_agent.set_mdp(mdp)

    if human_config["name"] == "greedy_agent":
        mlp_planner = MediumLevelPlanner(mdp, env_config["planner"])
        human_agent = GreedyHumanModel(
            mlp_planner, auto_unstuck=human_config["auto_unstuck"])
        human_agent.set_agent_index(1)
        human_agent.set_mdp(mdp)
    return ai_agent, human_agent, env


@wrap_experiment(snapshot_mode='last',
                 snapshot_gap=1,
                 name_parameters='all',
                 archive_launch_repo=False)
def rl_overcooked(ctxt=None, seed=np.random.randint(1000)):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    # set seed
    set_seed(seed)
    config["seed"] = seed

    # write config file to log directory
    with open(os.path.join(ctxt.snapshot_dir, "config.tml"), "w") as f:
        toml.dump(config, f)

    # config overcooked env and human agent
    ai_agent, human_agent, base_env = setup_env_w_agents(config)

    # get featurize_fn
    rl_config = config["RL"]
    featurize_fn = None
    featurize_fn_name = rl_config["featurize_fn"]
    if featurize_fn_name == "lossless_encoding":
        featurize_fn = lossless_state_featurize
    else:
        raise ValueError(
            f"Featurize function {featurize_fn_name} not supported.")

    # set up env and garage env
    overcooked_v1 = OvercookedV1(ai_agent, human_agent, base_env, featurize_fn)
    env = GymEnv(overcooked_v1, max_episode_length=base_env.horizon)

    # set up policy
    if featurize_fn_name == "lossless_encoding":
        policy = CategoricalCNNPolicy(
            env.spec,
            image_format="NHWC",
            kernel_sizes=(5, 5),
            hidden_channels=(10, 8),
        )

    # set up sampler
    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=base_env.horizon)

    # set up algo
    algo_name = rl_config["algo"]
    if algo_name == "ppo":
        value_function = GaussianCNNMLPValueFunction(
            env.spec,
            image_format="NHWC",
            kernel_sizes=(5, 5),
            hidden_channels=(10, 8),
        )

        batch_size = base_env.horizon * 10

        algo = PPO(env_spec=env.spec,
                   policy=policy,
                   sampler=sampler,
                   value_function=value_function,
                   discount=0.99,
                   gae_lambda=0.95,
                   lr_clip_range=0.2,
                   policy_optimizer=OptimizerWrapper(
                       (torch.optim.Adam, dict(lr=5e-4)),
                       policy,
                       max_optimization_epochs=64,
                       minibatch_size=batch_size // 64),
                   vf_optimizer=OptimizerWrapper(
                       (torch.optim.Adam, dict(lr=5e-4)),
                       value_function,
                       max_optimization_epochs=64,
                       minibatch_size=batch_size // 64),
                   entropy_method='max',
                   policy_ent_coeff=0.01,
                   center_adv=False,
                   stop_entropy_gradient=True,
                   use_softplus_entropy=False)

    else:
        raise ValueError(f"Algorithm {algo_name} is not supported.")

    trainer = Trainer(ctxt)
    trainer.setup(algo, env)
    trainer.train(n_epochs=rl_config["n_epochs"], batch_size=batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        help='path of config file',
                        required=True)
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = toml.load(f)

    rl_overcooked()
