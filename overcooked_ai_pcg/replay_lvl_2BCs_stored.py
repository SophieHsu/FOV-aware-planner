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

from overcooked_ai_pcg import LSI_CONFIG_EXP_DIR, LSI_LOG_DIR, LSI_CONFIG_ALGO_DIR, LSI_CONFIG_MAP_DIR, LSI_CONFIG_AGENT_DIR
from overcooked_ai_pcg.helper import run_overcooked_game, read_in_lsi_config
from overcooked_ai_pcg.LSI.qd_algorithms import Individual, FeatureMap
from overcooked_ai_pcg.helper import read_in_lsi_config, init_env_and_agent, visualize_lvl
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

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


# def retrieve_k_individuals(experiment_config, elite_map_config, agent_configs, individuals, row_idx, col_idx, log_dir, k):

#     #from IPython import embed
#     #embed()

#     num_simulations = experiment_config["num_simulations"
#     ]
#     feature_map = FeatureMap(
#         num_simulations,
#         feature_ranges=[(bc["low"], bc["high"])
#                         for bc in elite_map_config["Map"]["Features"]],
#         resolutions=[
#             bc["resolution"] for bc in elite_map_config["Map"]["Features"]
#         ],
#     )
#     feature0_name = features[0]["name"]
#     feature1_name = features[1]["name"]
#     #feature2_name = features[2]["name"]


#     num_individuals = num_simulations * 52
#     relevant_individuals = []
#     for indx in range(num_individuals):
#         individual = individuals.iloc[indx]
#         if indx % 1000== 0:
#           print(indx)

#         if (np.isnan(individual["ID"])==False): 
#              feature0 = individual[feature0_name]
#              feature1 = individual[feature1_name]
#              #feature2 = individual[feature2_name]
#              ind = Individual()
#              ind.features = ([feature0], [feature1])
#              #from IPython import embed
#              #em3bed()

#              ind.fitness = individual["fitness"]
#              index = feature_map.get_index(ind)

#              if index[0] == row_idx and index[1] == col_idx: 
#                 relevant_individuals.append(individual)

#     sorted_individuals = sorted(relevant_individuals, key=lambda x: x.fitness)[::-1]
#     sorted_individuals = sorted_individuals[:k]

#     IDs = []
#     for dd in range(k):
#       IDs.append(sorted_individuals[dd]["ID"])

    
#     return IDs, sorted_individuals


def play_individual(individual, agent_configs):
    ind = Individual()
    ind.level = individual.lvl_str
    ind.rand_seed = int(individual.rand_seed)
    agent1, agent2, env, mdp = init_env_and_agent(ind, agent_configs[-1])

    for agent_config in agent_configs:
        fitnesses, _, _, _, joint_actionses, concurr_actives, stuck_times = run_overcooked_game(ind, agent_config, render=True)
        print("Fitness: {}".format(fitnesses))
        print("Concurrently active:", concurr_actives, "; Stuck time:", stuck_times)
        #from IPython import embed
        #embed()
    return



def run_overcooked_local(agent1, agent2, env, mdp, render=True, worker_id=0, num_iters=1):
    """
    Run one turn of overcooked game and return the sparse reward as fitness
    """
    fitnesses = []; total_sparse_rewards = []; checkpointses = []; workloadses = []; 
    joint_actionses = []; concurr_actives = []; stuck_times = []
    np.random.seed(0)

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
                time.sleep(0.5)
            joint_action = (agent1.action(env.state)[0],
                            agent2.action(env.state)[0])
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

        #env = reset_env_from_mdp(mdp)

    #if agent_config["Search"]["multi_iter"] == False:
    fitnesses = [fitnesses[0] for i in range(num_iters)]
    total_sparse_rewards = [total_sparse_rewards[0] for i in range(num_iters)]
    checkpointses = [checkpointses[0] for i in range(num_iters)]
    workloadses = [workloadses[0] for i in range(num_iters)]
    joint_actionses = [joint_actionses[0] for i in range(num_iters)]
    concurr_actives = [concurr_actives[0] for i in range(num_iters)]
    stuck_times = [stuck_times[0] for i in range(num_iters)]

    # if num_iters > 1:
    #     checkpointses = np.array(checkpointses)
    #     fitnesses.append(median(fitnesses))
    #     total_sparse_rewards.append(sum(total_sparse_rewards)/len(total_sparse_rewards))
    #     checkpoint = [sum(checkpointses[:,i])/len(checkpointses[:,i]) for i in range(len(checkpointses[0]))]
    #     checkpointses = np.append(checkpointses, [checkpoint], axis=0)
    #     workloadses.append(get_workload_avg(workloadses))
    #     concurr_actives.append(np.median(concurr_actives))
    #     stuck_times.append(np.median(stuck_times))

    return fitnesses, total_sparse_rewards, checkpointses, workloadses, joint_actionses, concurr_actives, stuck_times



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


def play(elite_map, agent_configs, individuals, row_idx, col_idx, log_dir, load_data, agent_config_idx, mode):
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
        #curr_mat_idx = int(splited[2])

        ind_id = int(splited[2])

 
        if curr_row_idx == row_idx and curr_col_idx == col_idx:


            ind_id = int(splited[2])*51
            lvl_str = individuals["lvl_str"][ind_id]
            print("Playing in individual %d" % ind_id)
            print(lvl_str)




            ind = Individual()
            ind.level = lvl_str
            ind.human_preference = individuals["human_preference"][ind_id]
            ind.human_adaptiveness = individuals["human_adaptiveness"][ind_id]
            ind.rand_seed = int(individuals["rand_seed"][ind_id])

            log_file_name = str(row_idx) + "_" + str(col_idx) + "_" + str(agent_configs[agent_config_idx]['Agent1']['name'])


            #load_data_dir = os.path.join(log_dir, "stored_data/")
            #log_file_name = str(row_)

            load_data_dir = os.path.join(log_dir, "stored_data/")
            game_log = GameConfigLog(load_data_dir, log_file_name)

            if mode == "render":
              visualize_lvl(lvl_str, log_dir,f"rendered_level_{row_idx}_{col_idx}_{ind_id}.png")
            else:
               #from IPython import embed
               #embed()
                if load_data == 0: 
                    agent1, agent2, env, mdp = init_env_and_agent(ind, agent_configs[agent_config_idx])
                    game_log.addData(["agent1", "agent2", "mdp"], [agent1, agent2, mdp])
                    game_log.storeData()
                else:
                    agent1, agent2, mdp = game_log.loadData()
                    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=100)


               #game_log = GameConfigLog(load_data_dir, log_file_name)


                    #theApp = App(env, agent1, agent2, rand_seed=ind.rand_seed, player_idx=0, slow_time=False)
                    #ind.fitness, ind.scores, ind.checkpoints, ind.player_workloads, ind.joint_actions, ind.concurr_active, ind.stuck_time = theApp.on_execute()
                    for agent_config in agent_configs:
                        ind.fitness, ind.scores, ind.checkpoints, ind.player_workloads, ind.joint_actions, ind.concurr_active, ind.stuck_time  = run_overcooked_local(agent1, agent2, env, mdp, render=True)
                        print("Fitness: {}".format(ind.fitness))
                        print("Concurrently active:", ind.concurr_active, "; Stuck time:", ind.stuck_time)                #from IPython import embed
                        #embed()
                        #print("Fitness: %d" % fitness[0])
                        #log_actions(ind, agent_config, log_dir, f1, f2, row_idx, col_idx, ind_id)
                    #return

                    #print("Fitness: {}; Total sparse reward: {};".format(ind.fitness, ind.scores))
                    #print("Checkpoints", ind.checkpoints)
                    #print("Workloads:", ind.player_workloads, "; Concurrently active:", ind.concurr_active, "; Stuck time:", ind.stuck_time)

                    #log_actions(ind, agent_configs[-1], log_dir, f1, f2, row_idx, col_idx, ind_id)

            return

    print("No individual found in the specified cell")


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
    parser.add_argument('-row',
                        '--row_idx',
                        help='index f1 in elite map',
                        required=True)
    parser.add_argument('-col',
                        '--col_idx',
                        help='index f2 in elite map',
                        required=True)

    parser.add_argument('-id',
                        '--ind_id',
                        help='id of the individual',
                        required=False,
                        default=1)

    parser.add_argument('-load',
                        '--load_data',
                         help = 'load the data from pkl file',
                         required = False,
                         default = 0)


    parser.add_argument('-agent_config_idx',
                         '--agent_config_idx',
                         help='index of agent config',
                         required = False,
                         default = -1)


    parser.add_argument('-mode', '--mode',
                        help="mode to replay or merely render the level.",
                        required=False,
                        default="replay")


    opt = parser.parse_args()

    # read in full elite map
    log_dir = opt.log_dir
    elite_map_log_file = os.path.join(log_dir, "elite_map.csv")
    elite_map_log = open(elite_map_log_file, 'r')
    all_rows = list(csv.reader(elite_map_log, delimiter=','))
    elite_map = all_rows[-1][1:]
    elite_map_log.close()
    agent_config_idx = int(opt.agent_config_idx)

    load_data = int(opt.load_data)

    mode = opt.mode

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
    num_row = features[0]['resolution']
    num_col = features[1]['resolution']
    row_idx = int(opt.row_idx)
    col_idx = int(opt.col_idx)

    #from IPython import embed
    #embed()

    assert (row_idx < num_row)
    assert (col_idx < num_col)

    play(elite_map, agent_configs, individuals, row_idx, col_idx, log_dir, load_data, agent_config_idx, mode = mode)
 
    exit()

    #IDs, individuals  = retrieve_k_individuals(experiment_config, elite_map_config, agent_configs, individuals, row_idx, col_idx, log_dir,3)
    #for individual in individuals:
    #  play_individual(individual, agent_configs)

    #from IPython import embed
    #embed()
    



