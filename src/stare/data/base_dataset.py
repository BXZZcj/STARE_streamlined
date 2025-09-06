from abc import ABC, abstractmethod
import numpy as np
from dv import AedatFile
from typing import List, Dict


class BaseDataset(ABC):
    @abstractmethod
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.sequence_list = []
        # For lazy loading
        self.cur_loaded_seq_events_name = None
        self.cur_loaded_seq_rgb_name = None
    
    @abstractmethod
    def get_sequence_list(self):
        pass
    
    @abstractmethod
    def _get_sequence_events(self, sequence_name)->np.ndarray:
        pass

    @abstractmethod
    def _get_sequence_rgb(self, sequence_name)->np.ndarray:
        pass
    
    @abstractmethod
    def get_events_duration_sec(self, sequence_name:str)->float:
        pass

    @abstractmethod
    def _get_events_by_regular_duration(self, sequence_name:str, ts_start_sec:float, ts_end_sec:float)->np.ndarray:
        pass

    @abstractmethod
    def get_step_input_by_regular_duration(self, sequence_name:str, ts_start_sec:float, ts_end_sec:float)->np.ndarray:
        pass
    
    @abstractmethod
    def get_init_input_by_regular_duration(self, sequence_name:str, ts_start_sec:float, ts_end_sec:float)->np.ndarray:
        pass

    @abstractmethod
    def get_earlist_aval_init_input_exact_ts_sec(self, sequence_name:str, init_sampling_window_ms:float)->float:
        pass

    @abstractmethod
    def get_sequence_gt(self, sequence_name:str)->List[Dict]:
        pass