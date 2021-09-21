import argparse, toml, os
import collections
import random, time
import numpy as np
import dask.distributed
from dask_jobqueue import SLURMCluster

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import (HRLTrainingAgent, GreedyHumanModel, RandomAgent)
from overcooked_ai_py.planning.planners import (HumanSubtaskQMDPPlanner, MediumLevelPlanner)

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

def setup_env_w_agents(config, n_epi=None, env_list=None, human=None, q=None):
    """
    Setup environment and agents.

    NOTE: Green hat is agent 1/robot; Blue hat is agent 2/human
    """
    env_config, human_config = config["Env"], config["Human"]
    env_config["layout_dir"].split('/')[-1]
    if env_list is not None:
        if n_epi < len(env_list):
            env_config["layout_name"] = env_config["layout_dir"].split('/')[-1]+'/'+env_list[n_epi].split('.')[0]
        else:
            r = random.randint(0, len(env_list)-1)
            env_config["layout_name"] = env_config["layout_dir"].split('/')[-1]+'/'+env_list[r].split('.')[0]

    mdp = OvercookedGridworld.from_layout_name(
        env_config["layout_name"], **env_config["params_to_overwrite"])
    env = OvercookedEnv.from_mdp(mdp,
                                 info_level=0,
                                 horizon=env_config["horizon"])

    human_type = human_config["name"] if human is None else human

    if human_type == "mix_agent":
        # train with random_agent for every other 500 rounds
        flag = int(n_epi / 500) % 2
        if flag == 0:
            human_type = "random_agent"
        else:
            human_type = "greedy_agent"

    if human_type == "greedy_agent":
        mlp_planner = MediumLevelPlanner(mdp, env_config["planner"])
        human_agent = GreedyHumanModel(
            mlp_planner, auto_unstuck=human_config["auto_unstuck"])
        human_agent.set_agent_index(1)
        human_agent.set_mdp(mdp)

    elif human_type == "random_agent":
        human_agent = RandomAgent()
        human_agent.set_agent_index(1)
        human_agent.set_mdp(mdp)

    qmdp_planner = HumanSubtaskQMDPPlanner.from_pickle_or_compute(
            mdp, env_config["planner"], force_compute_all=True, info=False)
    ai_agent = HRLTrainingAgent(mdp, 
            qmdp_planner, 
            auto_unstuck=config["Robot"]["auto_unstuck"],
            qnet=q)
    ai_agent.set_agent_index(0)
    ai_agent.set_mdp(mdp)

    return ai_agent, human_agent, env, mdp

def reset(mdp, config):
    env_config = config["Env"]
    env = OvercookedEnv.from_mdp(mdp,
             info_level=0,
             horizon=env_config["horizon"])
    s = lstate_to_hstate(env.state)
    return s, env

def lstate_to_hstate(state, to_str=False):
    # hstate: subtask state and key object position
    objects = {'onion': 0, 'soup': 1, 'dish': 2, 'None': 3}

    player = state.players[0]
    other_player = state.players[1]
    player_obj = None; other_player_obj = None
    
    if player.held_object is not None:
        player_obj = player.held_object.name
    if other_player.held_object is not None:
        other_player_obj = other_player.held_object.name

    order_str = None if len(state.order_list) == 0 else state.order_list[0]
    for order in state.order_list[1:]:
        order_str = order_str + '_' + str(order)

    num_item_in_pot = 0
    if state.objects is not None and len(state.objects) > 0:
        for obj_pos, obj_state in state.objects.items():
            if obj_state.name == 'soup' and obj_state.state[1] > num_item_in_pot:
                num_item_in_pot = obj_state.state[1]

    state_vector = np.array([objects[str(player_obj)], num_item_in_pot, len(state.order_list[1:]), objects[str(other_player_obj)]])
    
    if to_str:
        state_strs = str(player_obj)+'_'+str(num_item_in_pot)+'_'+ order_str + '_' + str(other_player_obj)
        return np.array(state_vector), state_strs

    return np.array(state_vector)

def encode_env(mdp):
    onion_loc = mdp.get_onion_dispenser_locations()[0]
    pot_loc = mdp.get_pot_locations()[0]
    dish_loc = mdp.get_dish_dispenser_locations()[0]
    serving = mdp.get_serving_locations()[0]

    # create 4x2 matrix
    env_items = [[onion_loc[0], onion_loc[1]],
                [pot_loc[0], pot_loc[1]],
                [dish_loc[0], dish_loc[1]],
                [serving[0], serving[1]]]

    return np.array(env_items)

def haction_to_laction(env, ai_agent, state_strs, a_idx):
    laction = ai_agent.mdp_action_to_low_level_action(env.state, [state_strs], list(ai_agent.mdp_planner.subtask_dict.values())[a_idx])

    return laction

def step(env, h_state, ai_agent, human_agent, a_idx, reward_mode):
    s = h_state; s_prime = h_state; r_total = 0
    
    # loop until high-level state changes
    while (s == s_prime).all():
        h_state, h_state_strs = lstate_to_hstate(env.state, to_str=True)
        assert (s_prime == h_state).all(), 's_prime != h_state'
        
        ai_l_action = haction_to_laction(env, ai_agent, h_state_strs, a_idx)
        joint_action = (ai_l_action, human_agent.action(env.state)[0])
        l_s_prime, _, done, info = env.step(joint_action)
        s_prime = lstate_to_hstate(l_s_prime)

        # reward
        ai_shaped_reward = info["shaped_r_by_agent"][0]
        ai_sparse_reward = info["sparse_r_by_agent"][0]

        if reward_mode == "sparse":
            reward = ai_sparse_reward
        elif reward_mode == "shaped":
            reward = ai_shaped_reward
        elif reward_mode == "both":
            reward = ai_sparse_reward + ai_shaped_reward
        else:
            raise ValueError(f"Unknown reward mode: '{reward_mode}'.")
        r_total = r_total + reward

        if done and env.t < 400:
            break
        if done:
            break
        # if reward > 2:
        #     print(s, s_prime, reward)
    return s_prime, r_total, done, info

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)

    def put_trans_arr(self, trans_array):
        for i in trans_array:
            self.buffer.append(i)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s.flatten())
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime.flatten())
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(12, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(torch.flatten(obs))
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,4)
        else : 
            return out.argmax().item()
            
def init_dask(experiment_config, log_dir):
    """Initializes Dask with a local or SLURM cluster.

    Args:
        experiment_config (toml): toml config object of experiment
        log_dir (str): directory for storing logs
    Returns:
        A Dask client for the cluster created.
    """
    num_cores = experiment_config["num_cores"]

    if experiment_config.get("slurm", False):
        worker_logs = os.path.join(log_dir, "worker_logs")
        if not os.path.isdir(worker_logs):
            os.mkdir(worker_logs)
        output_file = os.path.join(worker_logs, 'slurm-%j.out')

        cores_per_worker = experiment_config["num_cores_per_slurm_worker"]

        # 1 process per CPU since cores == processes
        cluster = SLURMCluster(
            project=experiment_config["slurm_project"],
            cores=cores_per_worker,
            memory=f"{experiment_config['mem_gib_per_slurm_worker']}GiB",
            processes=cores_per_worker,
            walltime=experiment_config['slurm_worker_walltime'],
            job_extra=[
                f"--output {output_file}",
                f"--error {output_file}",
            ],
            death_timeout=3600,
        )

        print("### SLURM Job script ###")
        print("--------------------------------------")
        print(cluster.job_script())
        print("--------------------------------------")

        cluster.scale(cores=num_cores)
        return dask.distributed.Client(cluster)

    # Single machine -- run with num_cores worker processes.
    cluster = dask.distributed.LocalCluster(n_workers=num_cores,
                                            threads_per_worker=1,
                                            processes=True)
    return dask.distributed.Client(cluster)
        
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def run_env_w_agent(n_epi, q, env_list, eta):
    if config['Env']['multi']:
        ai_agent, human_agent, env, mdp = setup_env_w_agents(config, n_epi, env_list)

    epsilon = max(0.01, 0.5 - 0.01*(n_epi/eta)) #Linear annealing from 8% to 1%
    h_state, env = reset(mdp, config)
    env_items = encode_env(mdp)
    done = False
    trans_array = []
    score = 0.0
    while not done:
        # stack ai and human state info with environment info
        h_env_state = np.concatenate((h_state.reshape(4,1), env_items), axis=1)

        # sample action for robot
        a = q.sample_action(torch.from_numpy(h_env_state).float(), epsilon)      
        h_s_prime, r, done, info = step(env, h_state, ai_agent, human_agent, a, config["RL"]["reward_mode"])
        h_env_state_prime = np.concatenate((h_s_prime.reshape(4,1), env_items), axis=1)

        done_mask = 0.0 if done else 1.0
        trans_array.append((h_env_state, a, r/100.0, h_env_state_prime, done_mask))
        h_state = h_s_prime

        score += r
    
    return n_epi, trans_array, score, epsilon

def main(dask_client, config):
    # config overcooked env and human agent
    if not config['Env']['multi']:
        ai_agent, human_agent, env, mdp = setup_env_w_agents(config)
    else:
        env_list=os.listdir(config['Env']['layout_dir'])
        env_list.remove('base.layout')

    # initialize network
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = config['Experiment']['log_freq']
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    eta = config['RL']['eta']
    start_training_mem = config['RL']['memory_min']
    log_file = os.path.join(config["Experiment"]["log_dir"], config["Experiment"]["log_name"])

    if not os.path.exists(log_file):
        os.mkdir(log_file)
    with open(os.path.join(log_file, 'config.tml'), "w") as toml_file:
        toml.dump(config, toml_file)

    evaluations = []
    active_evals = 0
    n_epi = 0
    completed_evals = 0
    
    start_time = time.time()

    def _request_envs_w_agent(before_loop):
        nonlocal active_evals, n_epi
        while active_evals < config["num_cores"]:
            future = dask_client.submit(
                run_env_w_agent,
                n_epi, 
                q,
                env_list, 
                eta,
            )
            active_evals += 1
            if before_loop:
                evaluations.append(future)  # pylint: disable=no-member
            else:
                evaluations.add(future)
            print(f"{n_epi}/{config['RL']['n_epochs']}")
            n_epi += 1
        print(f"Active evaluations: {active_evals}")

    _request_envs_w_agent(True)
    evaluations = dask.distributed.as_completed(evaluations)
    print(f"Started {active_evals} simulations")        

    # completion time of the latest simulation
    last_eval = time.time()

    for completion in evaluations:
        # process the individual
        active_evals -= 1

        try:
            ind_n_epi, ind_trans_array, ind_score, epsilon  = completion.result()
            memory.put_trans_arr(ind_trans_array)
            score += ind_score

            if memory.size()>start_training_mem:
                train(q, q_target, memory, optimizer)

            if (ind_n_epi%print_interval==0 and ind_n_epi!=0) or (ind_n_epi == config['RL']['n_epochs']-1):
                q_target.load_state_dict(q.state_dict())
                print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(ind_n_epi, score/print_interval, memory.size(), epsilon*100))
                score = 0.0

                # save model
                torch.save(q.state_dict(), '{0}/qnet_epi_{1}.pth'.format(log_file, ind_n_epi))
                torch.save({
                    'epoch': ind_n_epi,
                    "q": q.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, '{0}/saved_model.tar'.format(log_file))
            
            completed_evals += 1

        except dask.distributed.scheduler.KilledWorker as err:  
            # worker may fail due to, for instance, memory
            print("Worker failed with the following error; continuing anyway\n"
                  "-------------------------------------------\n"
                  f"{err}\n"
                  "-------------------------------------------")
            # avoid not sending out more evaluations while evaluating initial
            # populations
            # algorithm.individuals_disbatched -= 1
            continue
        
        del completion

        if completed_evals < config['RL']['n_epochs']:
            # request more evaluations if still running
            _request_envs_w_agent(False)
        else:
            # otherwise, terminate
            break

        finish_time = time.time()
    print("Total time:", str(finish_time - start_time), "seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        help='path of config file',
                        required=True)
    parser.add_argument('-q',
                        '--qnet',
                        help='Qnet pth file',
                        default=None,
                        required=False)
    opt = parser.parse_args()

    n_eqi = None
    if opt.qnet is None:
        with open(opt.config) as f:
            config = toml.load(f)

    else:
        q_log_dir = opt.qnet.split("/")[:-1]
        with open(os.path.join(q_log_dir, 'config.tml')) as f:
            config = toml.load(f)

        q = Qnet()
        q.load_state_dict(torch.load(opt.qnet))
        q.eval()

        n_eqi = q_log_dir = opt.qnet.split("/")[-1].split(".")[0].split("_")[-1]

    log_file = os.path.join(config["Experiment"]["log_dir"], config["Experiment"]["log_name"])
    main(init_dask(config, log_file), config)
