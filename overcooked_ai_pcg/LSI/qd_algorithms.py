import numpy as np
import math
import os
import json
import toml
import pandas as pd
import csv
from abc import ABC, abstractmethod


class Individual:
    def __init__(self):
        pass

    def read_mario_features(self):
        pass

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
      if feature-1e-9 <= feature_range[0]:
         return 0
      if feature_range[1] <= feature + 1e-9:
         return self.resolutions[feature_id]-1

      gap = feature_range[1] - feature_range[0]
      pos = feature - feature_range[0]
      index = int((self.resolutions[feature_id] * pos + 1e-9) / gap)
      return index

   def get_index(self, cur):
      return tuple(self.get_feature_index(i, f) for i, f in enumerate(cur.features))

   def add_to_map(self, to_add):
      index = self.get_index(to_add)

      replaced_elite = False
      if index not in self.elite_map:
         self.elite_indices.append(index)
         self.elite_map[index] = to_add
         replaced_elite = True
         to_add.delta = (1, to_add.fitness)
      elif self.elite_map[index].fitness < to_add.fitness:
         to_add.delta = (0, to_add.fitness-self.elite_map[index].fitness)
         self.elite_map[index] = to_add
         replaced_elite = True

      return replaced_elite

   def add(self, to_add):
      self.num_individuals_added += 1
      replaced_elite = self.add_to_map(to_add)
      return replaced_elite

   def get_random_elite(self):
      pos = np.random.randint(0, len(self.elite_indices)-1)
      index = self.elite_indices[pos]
      return self.elite_map[index]


class QDAlgorithmBase(ABC):
    @abstractmethod
    def is_running(self):
        pass

    @abstractmethod
    def generate_individual(self):
        pass

    @abstractmethod
    def return_evaluated_individual(self):
        pass


class MapElitesAlgorithm(QDAlgorithmBase):
    def __init__(self,
                 mutation_power,
                 initial_population,
                 num_to_evaluate,
                 feature_map,
                 num_params=32):
        super().__init__()
        self.num_to_evaluate = num_to_evaluate
        self.initial_population = initial_population
        self.individuals_evaluated = 0
        self.feature_map = feature_map
        self.mutation_power = mutation_power
        self.individuals_disbatched = 0
        self.num_params = num_params

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def generate_individual(self):
        ind = Individual()
        if self.individuals_disbatched < self.initial_population:
            ind.param_vector = np.random.normal(0.0, 1.0, self.num_params)
        else:
            parent = self.feature_map.get_random_elite()
            ind.param_vector = parent.param_vector
            + np.random.normal(0.0, self.mutation_power, self.num_params)
        self.individuals_disbatched += 1
        return ind

    def return_evaluated_individual(self, ind):
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1
        self.feature_map.add(ind)

class RandomGenerator(QDAlgorithmBase):
    def __init__(self,
                 num_to_evaluate,
                 feature_map,
                 num_params=32):
        super().__init__()
        self.num_to_evaluate = num_to_evaluate
        self.individuals_evaluated = 0
        self.feature_map = feature_map
        self.num_params = num_params

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def generate_individual(self):
        ind = Individual()
        unscaled_params = np.random.normal(0.0, 1.0, self.num_params)
        ind.param_vector = unscaled_params
        return ind

    def return_evaluated_individual(self, ind):
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1
        self.feature_map.add(ind)