"""Quality-Diversity algorithms and their components."""
import dataclasses
from abc import ABC, abstractmethod

import numpy as np
import ray


@dataclasses.dataclass
class Individual:
    """Data for a single individual in a QD algorithm."""
    features = None  # BC's
    param_vector = None  # genotype
    level = None  # an Overcooked game level
    fitness = None  # fitness in the level = score - timestep
    score = None # raw score of the level,
                 # proportional to the number of soup delivered
    timestep = None # timestep to finish the level
    ID = None  # ID of the individual after being inserted to the map
    player_workload = None # list of dic that summarize workload of all players

class FeatureMap:

    def __init__(self, max_individuals, feature_ranges, resolutions):
        self.max_individuals = max_individuals
        self.feature_ranges = feature_ranges
        self.resolutions = resolutions

        self.elite_map = {}
        self.elite_indices = []

        self.num_individuals_added = 0

    def get_feature_index(self, feature_id, feature):
        feature_range = self.feature_ranges[feature_id]
        if feature - 1e-9 <= feature_range[0]:
            return 0
        if feature_range[1] <= feature + 1e-9:
            return self.resolutions[feature_id] - 1

        gap = feature_range[1] - feature_range[0]
        pos = feature - feature_range[0]
        index = int((self.resolutions[feature_id] * pos + 1e-9) / gap)
        return index

    def get_index(self, cur):
        return tuple(
            self.get_feature_index(i, f) for i, f in enumerate(cur.features))

    def add_to_map(self, to_add):
        index = self.get_index(to_add)

        replaced_elite = False
        if index not in self.elite_map:
            self.elite_indices.append(index)
            self.elite_map[index] = to_add
            replaced_elite = True
            to_add.delta = (1, to_add.fitness)
        elif self.elite_map[index].fitness < to_add.fitness:
            to_add.delta = (0, to_add.fitness - self.elite_map[index].fitness)
            self.elite_map[index] = to_add
            replaced_elite = True

        return replaced_elite

    def add(self, to_add):
        self.num_individuals_added += 1
        replaced_elite = self.add_to_map(to_add)
        return replaced_elite

    def get_random_elite(self):
        pos = np.random.randint(0, len(self.elite_indices) - 1)
        index = self.elite_indices[pos]
        return self.elite_map[index]


class QDAlgorithmBase(ABC):
    """Base class for all QD algorithms.

    Args:
        feature_map (FeatureMap): A container for storing solutions.
        running_individual_log (RunningIndividualLog): Previously constructed
            logger.
        frequent_map_log (FrequentMapLog): Previously constructed logger.
        map_summary_log (MapSummaryLog): Previously constructed logger.
    """

    def __init__(self, feature_map, running_individual_log, frequent_map_log,
                 map_summary_log):
        self.feature_map = feature_map
        self.individuals_disbatched = 0
        self.individuals_evaluated = 0
        self.running_individual_log = running_individual_log
        self.frequent_map_log = frequent_map_log
        self.map_summary_log = map_summary_log

    @abstractmethod
    def is_running(self):
        pass

    @abstractmethod
    def generate_individual(self):
        pass

    @abstractmethod
    def return_evaluated_individual(self, ind):
        pass

    def generate_individual_if_still_running(self):
        """
        If the algorithm is still running, generates an individual and returns
        it with its id.

        The id can be as simple as the number of individuals that have been
        generated so far (this is its default value, in fact).

        Returns:
            ind (Individual): The generated individual. None if the algorithm is
                no longer running.
            ind_id (int): The id of the generated individual. None if the
                algorithm is no longer running.
        """
        if self.is_running():
            return self.generate_individual(), self.individuals_disbatched
        return None, None

    def insert_if_still_running(self, ind):
        """
        Adds the individual to the algorithm container if the algorithm is still
        running and logs the results.

        Returns:
            running (bool): Whether the algorithm is still running.
            individuals_evaluated (int): The number of individuals that have
                been evaluated so far. None if the algorithm is no longer
                running.
        """
        if self.is_running():
            evaluated_ind = self.return_evaluated_individual(ind)
            self.running_individual_log.log_individual(evaluated_ind)
            self.frequent_map_log.log_map(self.feature_map)
            self.map_summary_log.log_summary(self.feature_map,
                                             self.individuals_evaluated)
            return True, self.individuals_evaluated
        return False, None


@ray.remote
class MapElitesAlgorithm(QDAlgorithmBase):

    def __init__(self,
                 mutation_power,
                 initial_population,
                 num_to_evaluate,
                 feature_map,
                 running_individual_log,
                 frequent_map_log,
                 map_summary_log,
                 num_params=32):
        super().__init__(feature_map, running_individual_log, frequent_map_log,
                         map_summary_log)
        self.num_to_evaluate = num_to_evaluate
        self.initial_population = initial_population
        self.mutation_power = mutation_power
        self.num_params = num_params

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def generate_individual(self):
        ind = Individual()
        if self.individuals_disbatched < self.initial_population:
            ind.param_vector = np.random.normal(0.0, 1.0, self.num_params)
        else:
            parent = self.feature_map.get_random_elite()
            ind.param_vector = parent.param_vector + np.random.normal(
                0.0, self.mutation_power, self.num_params)
        self.individuals_disbatched += 1
        return ind

    def return_evaluated_individual(self, ind):
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1
        self.feature_map.add(ind)
        return ind


@ray.remote
class RandomGenerator(QDAlgorithmBase):

    def __init__(self,
                 num_to_evaluate,
                 feature_map,
                 running_individual_log,
                 frequent_map_log,
                 map_summary_log,
                 num_params=32):
        super().__init__(feature_map, running_individual_log, frequent_map_log,
                         map_summary_log)
        self.num_to_evaluate = num_to_evaluate
        self.num_params = num_params

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def generate_individual(self):
        ind = Individual()
        unscaled_params = np.random.normal(0.0, 1.0, self.num_params)
        ind.param_vector = unscaled_params
        self.individuals_disbatched += 1
        return ind

    def return_evaluated_individual(self, ind):
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1
        self.feature_map.add(ind)
        return ind
