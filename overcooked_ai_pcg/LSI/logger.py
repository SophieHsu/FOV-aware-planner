"""Contains logger classes and a Ray actor for bundling all of them together."""
import csv
import os
from abc import ABC, abstractmethod

import numpy as np
import ray
from overcooked_ai_pcg import LSI_LOG_DIR


class LoggerBase(ABC):

    @abstractmethod
    def init_log(self,):
        pass

    def _write_row(self, to_add):
        """Append a row to csv file"""
        with open(self._log_path, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(to_add)
            f.close()


class RunningIndividualLog(LoggerBase):
    """
    Logger that logs individuals to a csv file

    Args:
        log_path (string): filename of the log file
        elite_map_config: toml config object of the feature maps
    """

    def __init__(self, log_path, elite_map_config):
        super().__init__()
        self._log_path = os.path.join(LSI_LOG_DIR, log_path)
        self._isInitialized = False
        self._elite_map_config = elite_map_config
        self.init_log()

    def init_log(self,):
        # remove the file if exists
        if os.path.exists(self._log_path):
            os.remove(self._log_path)

        # construct labels
        data_labels = ["ID", "fitness", "score", "timestep"]
        for bc in self._elite_map_config["Map"]["Features"]:
            data_labels.append(bc["name"])
        data_labels.append("lvl_str")
        self._write_row(data_labels)

    def log_individual(self, ind):
        to_add = [
            ind.ID,
            ind.fitness,
            ind.score,
            ind.timestep,
            *ind.features,
            ind.level,
        ]
        self._write_row(to_add)


class FrequentMapLog(LoggerBase):
    """
    Logger that logs compressed feature map information

    Args:
        log_path (string): filename of the log file
        num_features (int): number of behavior characteristics
    """

    def __init__(self, log_path, num_features):
        super().__init__()
        self._log_path = os.path.join(LSI_LOG_DIR, log_path)
        self.init_log(num_features)

    def init_log(self, num_features):
        # remove the file if exists
        if os.path.exists(self._log_path):
            os.remove(self._log_path)

        # construct label
        feature_label = ":".join(
            ["feature" + str(i) for i in range(num_features)])
        data_labels = [
            "Dimension", "f1:f2:IndividualID:Fitness:" + feature_label
        ]
        self._write_row(data_labels)

    def log_map(self, feature_map):
        to_add = []
        to_add.append("x".join(str(num) for num in feature_map.resolutions),)
        for index in feature_map.elite_indices:
            ind = feature_map.elite_map[index]
            curr = [
                *index,
                ind.ID,
                ind.fitness,
                *ind.features,
            ]
            to_add.append(":".join(str(ele) for ele in curr))
        self._write_row(to_add)


class MapSummaryLog(LoggerBase):
    """
    Logger that logs general feature map info to a csv file

    Args:
        log_path (string): filename of the log file
    """

    def __init__(self, log_path):
        super().__init__()
        self._log_path = os.path.join(LSI_LOG_DIR, log_path)
        self.init_log()

    def init_log(self,):
        # remove the file if exists
        if os.path.exists(self._log_path):
            os.remove(self._log_path)

        data_labels = [
            "NumEvaluated",
            "QD-Score",
            "MeanNormFitness",
            "MedianNormFitness",
            "MaxNormFitness",
            "CellsOccupied",
            "PercentOccupied",
        ]
        self._write_row(data_labels)

    def log_summary(self, feature_map, num_evaluated):
        all_fitness = []
        for index in feature_map.elite_indices:
            ind = feature_map.elite_map[index]
            all_fitness.append(ind.fitness)
        cells_occupied = len(feature_map.elite_indices)
        QD_score = np.sum(all_fitness)
        mean_fitness = np.average(all_fitness)
        median_fitness = np.median(all_fitness)
        max_fitness = np.max(all_fitness)
        num_cell = np.prod(feature_map.resolutions)
        percent_occupied = 100 * cells_occupied / num_cell

        to_add = [
            num_evaluated,
            QD_score,
            mean_fitness,
            median_fitness,
            max_fitness,
            cells_occupied,
            percent_occupied,
        ]
        self._write_row(to_add)
