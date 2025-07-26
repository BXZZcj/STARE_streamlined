from abc import ABC, abstractmethod
import numpy as np
from dv import AedatFile


class BaseDataset(ABC):
    @abstractmethod
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.sequence_list = []
    
    @abstractmethod
    def get_sequence_list(self):
        pass
    
    @abstractmethod
    def get_sequence_events(self, sequence_name):
        pass

    @abstractmethod
    def get_sequence_gt(self, sequence_name):
        pass