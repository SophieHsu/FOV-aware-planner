import os
import csv
import time
import json
import toml
import argparse
import pandas as pd
import pygame
import numpy as np
import pickle 
import random

from overcooked_ai_pcg import LSI_CONFIG_EXP_DIR, LSI_LOG_DIR, LSI_CONFIG_ALGO_DIR, LSI_CONFIG_MAP_DIR, LSI_CONFIG_AGENT_DIR
from overcooked_ai_pcg.helper import run_overcooked_game, read_in_lsi_config
from overcooked_ai_pcg.LSI.qd_algorithms import Individual, FeatureMap
from overcooked_ai_pcg.helper import read_in_lsi_config, init_env_and_agent
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_pcg.LSI.bc_calculate import diff_num_ingre_held
# def read_in_lsi_config(exp_config_file):

# def read_in_lsi_config(exp_config_file):
#     experiment_config = toml.load(exp_config_file)
#     algorithm_config = toml.load(
#         os.path.join(LSI_CONFIG_ALGO_DIR,
#                      experiment_config["experiment_config"]["algorithm_config"]))
#     elite_map_config = toml.load(
#         os.path.join(LSI_CONFIG_MAP_DIR,
#                      experiment_config["experiment_config"]["elite_map_config"]))
#     agent_configs = []
#     for agent_config_file in experiment_config["agent_config"]:
#         agent_config = toml.load(
#         os.path.join(LSI_CONFIG_AGENT_DIR, experiment_config["experiment_config"]["agent_config"]))
#         agent_configs.append(agent_config)
#     return experiment_config, algorithm_config, elite_map_config, agent_configs


class GameConfigLog:
    def __init__(self, log_dir, filename):
        self.db = {}
        self.full_path = os.path.join(log_dir, filename+'.pkl')

    def addData(self, keys, values):
        for key, value in zip(keys, values):
            self.db[key] = value 

    def storeData(self): 
        dbfile = open(self.full_path, 'wb') 
        pickle.dump(self.db, dbfile)                      
        dbfile.close() 
      
    def loadData(self): 
        # for reading also binary mode is important 
        dbfile = open(self.full_path, 'rb')      
        self.db = pickle.load(dbfile) 
        agent1 = self.db['agent1']
        agent2 = self.db['agent2']
        mdp = self.db['mdp']
        dbfile.close() 

        return agent1, agent2, mdp

    def is_exist(self):
        return os.path.exists(self.full_path)

def reset_env_from_mdp(mdp):
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=100)
    return env

def run_overcooked_local(agent1, agent2, env, mdp, eps = 0, render=True, worker_id=0, num_iters=50):
    """
    Run one turn of overcooked game and return the sparse reward as fitness
    """

    fitnesses = []; total_sparse_rewards = []; checkpointses = []; workloadses = []; 
    joint_actionses = []; concurr_actives = []; stuck_times = []
    #np.random.seed(0)

    for num_iter in range(num_iters):
        done = False
        total_sparse_reward = 0
        last_state = None
        timestep = 0

        # Saves when each soup (order) was delivered
        checkpoints = [env.horizon - 1] * env.num_orders
        cur_order = 0

        # store all actions
        joint_actions = []

        while not done:
            if render:
                env.render()
                time.sleep(1.0)

            joint_action = (agent1.action(env.state)[0],
                            agent2.action(env.state, eps)[0])
            # action1 = agent1.action(env.state)[0]
            # if random.random() < 1:
            #   action2 = (Action.ALL_ACTIONS[np.random.randint(6)],{})[0]
            # else: 
            #   action2 = agent2.action(env.state)[0]

            #joint_action = (action1,action2)

            # print(joint_action)
            joint_actions.append(joint_action)
            next_state, timestep_sparse_reward, done, info = env.step(joint_action)
            total_sparse_reward += timestep_sparse_reward

            if timestep_sparse_reward > 0:
                checkpoints[cur_order] = timestep
                cur_order += 1




            last_state  = next_state
            timestep += 1


        workloads = last_state.get_player_workload()
        concurr_active = last_state.cal_concurrent_active_sum()
        stuck_time = last_state.cal_total_stuck_time()

        # Smooth fitness is the total reward tie-broken by soup delivery times.
        # Later soup deliveries are higher priority.
        fitness = total_sparse_reward + 1
        for timestep in reversed(checkpoints):
            fitness *= env.horizon
            fitness -= timestep


        #print("fitness is: " + str(fitness))

        fitnesses.append(fitness)
        total_sparse_rewards.append(total_sparse_reward)
        checkpointses.append(checkpoints)
        workloadses.append(workloads)
        joint_actionses.append(joint_actions)
        concurr_actives.append(concurr_active)
        stuck_times.append(stuck_time)

        env = reset_env_from_mdp(mdp)


    # if num_iters > 1:
    #     checkpointses = np.array(checkpointses)
    #     fitnesses.append(np.median(fitnesses))
    #     total_sparse_rewards.append(sum(total_sparse_rewards)/len(total_sparse_rewards))
    #     checkpoint = [sum(checkpointses[:,i])/len(checkpointses[:,i]) for i in range(len(checkpointses[0]))]
    #     checkpointses = np.append(checkpointses, [checkpoint], axis=0)
    #     workloadses.append(get_workload_avg(workloadses))
    #     concurr_actives.append(np.median(concurr_actives))
    #     stuck_times.append(np.median(stuck_times))


    return fitnesses, total_sparse_rewards, checkpointses, workloadses, joint_actionses, concurr_actives, stuck_times

def play(elite_map, agent_configs, individuals, log_dir, agent_config_idx, BC_dim, eps):
    """
    Find the individual in the specified cell in the elite map
    and run overcooked game with the specified agents
    Args:
        elite_map (list): list of logged cell strings.
                          See elite_map.csv for detail.
        agent_configs: toml config object of agents
        individuals (pd.dataFrame): all individuals logged
        f1, f2 (int, int): index of the features to use
        row_idx, col_idx (int, int): index of the cell in the elite map
    """



    for elite in elite_map:
        splited = elite.split(":")
        curr_row_idx = int(splited[0])
        curr_col_idx = int(splited[1])

        ind_id = int(splited[2])

 
        ind_id = int(splited[2])*51
        lvl_str = individuals["lvl_str"][ind_id]
        print("Playing in individual %d" % ind_id)
        print(lvl_str)

            #from IPython import embed
            #embed()


        ind = Individual()
        ind.level = lvl_str
        #ind.human_preference = individuals["human_preference"][ind_id]
        #ind.human_adaptiveness = individuals["human_adaptiveness"][ind_id]
        ind.rand_seed = int(individuals["rand_seed"][ind_id])

        log_file_name = str(curr_row_idx) + "_" + str(curr_col_idx) +"_" + str(agent_configs[agent_config_idx]['Agent1']['name'])
         
        load_data_dir = os.path.join(log_dir, "stored_data_all/")
        BC_dir = os.path.join(log_dir, "BCs_" + str(int(eps*100))+"/" + str(BC_dim)+"/")
        if BC_dim == 0:
          file = open(BC_dir + str(curr_row_idx)+".dat", "a+")
        elif BC_dim == 1:
          file = open(BC_dir + str(curr_col_idx)+".dat", "a+")

        game_log = GameConfigLog(load_data_dir, log_file_name)


        agent1, agent2, mdp = game_log.loadData()
        env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=100)


        #theApp = App(env, agent1, agent2, rand_seed=ind.rand_seed, player_idx=0, slow_time=False)
        #ind.fitness, ind.scores, ind.checkpoints, ind.player_workloads, ind.joint_actions, ind.concurr_active, ind.stuck_time = theApp.on_execute()
        #for agent_config in agent_configs:
        #fitness, _, _, _, ind.joint_actions, _, _ = run_overcooked_local(agent1, agent2, env, mdp, render=True)
        #theApp = App(env, agent1, agent2, rand_seed=ind.rand_seed, player_idx=0, slow_time=False)
        #ind.fitness, ind.scores, ind.checkpoints, ind.player_workloads, ind.joint_actions, ind.concurr_active, ind.stuck_time = theApp.on_execute()
        ind.fitness, ind.scores, ind.checkpoints, ind.player_workloads, ind.joint_actions, ind.concurr_active, ind.stuck_time= run_overcooked_local(agent1, agent2, env, mdp, eps, render=False)
        print("Fitness: {}".format(ind.fitness))
        print("Concurrently active:", ind.concurr_active, "; Stuck time:", ind.stuck_time)                #from IPython import embed
         
        #feature_name = "num_ingre_held"
        #feature_val = ind.player_workloads[0][0]["num_ingre_held"]-ind.player_workloads[0][1]["num_ingre_held"]

  
 

            #feature_val = diff_num_ingre_held(ind)
        if BC_dim == 0: 
            file.writelines("%s\n" % place for place in ind.concurr_active)
        elif BC_dim == 1:
            file.writelines("%s\n" % place for place in ind.stuck_time)
    

        #file.write(str(workload_diff)+"\n")
        #from IPython import embed
        #embed()
            #print("Fitness: %d" % fitness[0])
            #log_actions(ind, agent_config, log_dir, f1, f2, row_idx, col_idx, ind_id)
        #return

        #print("Fitness: {}; Total sparse reward: {};".format(ind.fitness, ind.scores))
        #print("Checkpoints", ind.checkpoints)
        #print("Workloads:", ind.player_workloads, "; Concurrently active:", ind.concurr_active, "; Stuck time:", ind.stuck_time)

        #log_actions(ind, agent_configs[-1], log_dir, f1, f2, row_idx, col_idx, ind_id)

    return

    #print("No individual found in the specified cell")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        help='path of experiment config file',
                        required=True)
    parser.add_argument('-l',
                        '--log_dir',
                        help='path of log directory',
                        required=True)


    parser.add_argument('-eps', '--eps_value', help = 'probability of human random action', required = True)


    parser.add_argument('-agent_config_idx',
                         '--agent_config_idx',
                         help='index of agent config',
                         required = False,
                         default = -1)

    parser.add_argument('-b', '--BC_dim', help = 'BC to compute', required = False, default = 0)


    opt = parser.parse_args()

    # read in full elite map
    log_dir = opt.log_dir
    elite_map_log_file = os.path.join(log_dir, "elite_map.csv")
    elite_map_log = open(elite_map_log_file, 'r')
    all_rows = list(csv.reader(elite_map_log, delimiter=','))
    elite_map = all_rows[-1][1:]
    elite_map_log.close()
    agent_config_idx = int(opt.agent_config_idx)

    BC_dim = int(opt.BC_dim)
    eps = float(opt.eps_value)

    # read in individuals
    individuals = pd.read_csv(os.path.join(log_dir, "individuals_log.csv"))


    # read in configs
    experiment_config, _, elite_map_config, agent_configs = read_in_lsi_config(opt.config)

    # read in feature index
    features = elite_map_config['Map']['Features']
    #num_features = len(features)
    #f1 = int(opt.feature1_idx)
    #f2 = int(opt.feature2_idx)
    #assert (f1 < num_features)
    #assert (f2 < num_features)

    # read in row/col index
    #num_row = features[0]['resolution']
    #num_col = features[1]['resolution']
    #num_mat = features[2]['resolution']
    #row_idx = int(opt.row_idx)
    #col_idx = int(opt.col_idx)
    #mat_idx = int(opt.mat_idx)

    random.seed(0)
    sampled_map = random.sample(elite_map, 100)

    #assert (row_idx < num_row)
    #assert (col_idx < num_col)
    #assert (mat_idx < num_mat)
    #from IPython import embed
    #embed()

    play(sampled_map, agent_configs, individuals, log_dir, agent_config_idx, BC_dim, eps)
 
    exit()

    #IDs, individuals  = retrieve_k_individuals(experiment_config, elite_map_config, agent_configs, individuals, row_idx, col_idx, mat_idx, log_dir,3)
    #for individual in individuals:
    #  play_individual(individual, agent_configs)

    #from IPython import embed
    #embed()
    



