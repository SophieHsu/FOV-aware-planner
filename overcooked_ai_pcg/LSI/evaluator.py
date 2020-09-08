import time
from multiprocessing import Process

from overcooked_ai_pcg.gen_lvl import generate_lvl, generate_rnd_lvl
from overcooked_ai_pcg.LSI import bc_calculate
from overcooked_ai_pcg.helper import run_overcooked_game

class Status:
    IDLE = "idle"
    EVALUATING = "evaluating"
    TERMINATING = "terminating"

class Worker(object):
    """
    Object used for multiprocess data passing

    Note: the status string would alternate between "idle", "evaluating", and "terminating" indicating the status of the process.
    """
    def __init__(self, id):
        self._id = id
        self._status = Status.IDLE

    def get_id(self):
        return self._id

    def get_status(self):
        return self._status
    
    def set_status(self, status):
        self._status = status

    def get_ind(self):
        return self._ind

    def set_ind(self,ind):
        self._ind = ind

    def get_sim_id(self):
        return self._sim_id

    def set_sim_id(self, sim_id):
        self._sim_id = sim_id

class OvercookedEvaluator(Process):
    """
    A worker process class that handles multiprocessed overcooked evaluation

    Args:
        id(int): id of the process
        obj(Worker): Worker obj managed by multiprocessing.manager used for multiprocess communication
        visualize (bool): render the game(evaluation) or not
        elite_map_config: toml config object of the feature maps
        model_path (string): file path to the GAN model
    """
    def __init__(self, id, msg, visualize, elite_map_config, model_path):
        self.id = id
        self.msg = msg
        self.visualize = visualize
        self.elite_map_config = elite_map_config
        self.model_path = model_path
        super(OvercookedEvaluator, self).__init__()

    def run(self):
        while(True):
            if self.msg.get_status() == Status.EVALUATING:
                ind = self.msg.get_ind()
                evaluated_ind = self.eval_overcooked(ind)
                self.msg.set_ind(evaluated_ind)
                self.msg.set_status(Status.IDLE)
            elif self.msg.get_status() == Status.IDLE:
                pass
                # print("Process %d waiting..." % self.id)
            elif self.msg.get_status() == Status.TERMINATING:
                return
            time.sleep(1)

    def eval_overcooked(self, ind):
        """
        Evaluate overcooked game by running a game and calculate relevant bc

        Args:
            ind (Individual): individual instance
        """
        # generate new level
        ind.level = generate_lvl(1, self.model_path,
                                 ind.param_vector,
                                 worker_id=self.id)
        # ind.level = generate_rnd_lvl((6, 8), worker_id=self.id)

        # run simulation
        ind.fitness = run_overcooked_game(ind.level,
                                          render=self.visualize,
                                          worker_id=self.id)

        # calculate bc out of the game
        ind.features = []
        for bc in self.elite_map_config["Map"]["Features"]:
            # get the function the calculate bc
            bc_fn_name = bc["name"]
            bc_fn = getattr(bc_calculate, bc_fn_name)
            bc_val = bc_fn(ind)
            ind.features.append(bc_val)
        ind.features = tuple(ind.features)
        print("worker_id(%d): Game end; fitness = %d" % (self.id, ind.fitness))
        return ind