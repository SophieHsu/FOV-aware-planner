"""Defines a Ray remote function for running evaluations."""
import gc
import resource

from overcooked_ai_pcg.GAN_training import dcgan
from overcooked_ai_pcg.gen_lvl import DocplexFailedError, generate_lvl
from overcooked_ai_pcg.helper import run_overcooked_game
from overcooked_ai_pcg.LSI import bc_calculate


def print_mem_usage(info, worker_id):
    print(f"worker({worker_id}): Memory usage ({info}):",
          resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def run_overcooked_eval(ind, visualize, elite_map_config, agent_configs,
                        G_params, gan_state_dict, worker_id):
    """
    Evaluates overcooked game by running a game and calculating relevant BC's.

    Args:
        ind (overcooked_ai_pcg.LSI.qd_algorithms.Individual): Individual
            generated by the algorithm for evaluate.
        visualize (bool): render the game(evaluation) or not
        elite_map_config: toml config object of the feature maps
        G_params: parameters for the GAN
        gan_state_dict: weights of the GAN
        worker_id (int): Worker ID to use.

    Returns:
        The individual that was passed in (ind), but with populated data fields.
    """
    print_mem_usage("start", worker_id)

    generator = dcgan.DCGAN_G(**G_params)
    generator.load_state_dict(gan_state_dict)

    print_mem_usage("after loading GAN", worker_id)

    # generate new level
    try:
        ind.level = generate_lvl(
            1,
            generator,
            # since this vector originates from the algorithm actor, Ray makes it
            # read-only; thus, we should copy it so generate_lvl can do whatever it
            # wants with it
            ind.param_vector[:32].copy(),
            worker_id=worker_id,
        )
    except DocplexFailedError:
        print("worker(%d): The Docplex subprocess seems to have failed." %
              worker_id)
        return None

    # ind.level = generate_rnd_lvl((6, 8), worker_id=self.id)
    del generator

    fitnesses = []
    scores = []
    checkpoints = []
    player_workloads = []
    joint_actions = []
    concurr_actives = []
    stuck_times = []

    # run evaluation for all sets of agent configs
    # normalize the fitness if more than 1 sets are running
    for agent_config in agent_configs:
        if agent_config["Search"]["type"] == 'human':
            # generate human worker preference and adaptiveness
            ind.human_preference = ind.param_vector[32]
            ind.human_adaptiveness = ind.param_vector[33]

        print_mem_usage("after generating level", worker_id)

        # run simulation
        try:
            fitness, score, checkpoint, workload, joint_action, concurr_active, stuck_time = run_overcooked_game(
                ind, agent_config, render=visualize, worker_id=worker_id)
            fitnesses.append(fitness)
            scores.append(score)
            checkpoints.append(checkpoint)
            player_workloads.append(workload)
            joint_actions.append(joint_action)
            concurr_actives.append(concurr_active)
            stuck_times.append(stuck_time)
        except TimeoutError:
            print(
                "worker(%d): Level generated taking too much time to plan. Skipping"
                % worker_id)
            return None

        # run the garbage collector after each simulation
        gc.collect()

    ind.fitnesses = tuple(fitnesses)
    ind.scores = tuple(scores)
    ind.checkpoints = tuple(checkpoints)
    ind.player_workloads = tuple(player_workloads)
    ind.joint_actions = tuple(joint_actions)
    ind.concurr_actives = tuple(concurr_actives)
    ind.stuck_times = tuple(stuck_times)

    # for unnormalized version, fitness is scale
    if len(agent_configs) == 1:
        ind.fitness = ind.fitnesses[0]
    # for normalized version, fitness is the difference between two runs
    elif len(agent_configs) == 2:
        ind.fitness = ind.fitnesses[0] - ind.fitnesses[1]

    print_mem_usage("after running overcooked game", worker_id)

    # calculate bc out of the game
    worker_id, ind = calculate_bc(worker_id, ind, elite_map_config)

    print_mem_usage("end", worker_id)
    gc.collect()
    return ind


def calculate_bc(worker_id, ind, elite_map_config):
    ind.features = []
    for bc in elite_map_config["Map"]["Features"]:
        # get the function that calculates bc
        bc_fn_name = bc["name"]
        bc_fn = getattr(bc_calculate, bc_fn_name)
        bc_val = bc_fn(ind)
        ind.features.append(bc_val)
    ind.features = tuple(ind.features)
    print("worker(%d): Game end; fitness = %d" % (worker_id, ind.fitness))

    return worker_id, ind