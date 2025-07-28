import os
import numpy as np
from dv import AedatFile
from typing import List, Dict
import torch

from .base_dataset import BaseDataset


class DemoVOTDataset(BaseDataset):
    def __init__(self, dataset_path:str):
        super().__init__(dataset_path)
        
        with open(os.path.join(dataset_path, 'test.txt'), 'r') as f:
            self.sequence_list = f.read().splitlines()

    def get_sequence_list(self):
        return self.sequence_list
    
    def get_sequence_events(self, sequence_name)->np.ndarray:
        aedat4_path = os.path.join(self.dataset_path, "aedat4", f'{sequence_name}.aedat4')
        
        with AedatFile(aedat4_path) as f:
            events = np.hstack([packet for packet in f['events'].numpy()])
            events['timestamp'] = events['timestamp'] - events['timestamp'][0]
            
        return events
    
    def get_sequence_gt(self, sequence_name:str)->List[Dict]:
        gt_path = os.path.join(self.dataset_path, "annots", f'{sequence_name}.txt')
        ground_truth = []
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.split()
                timestamp = float(parts[0])
                bbox = torch.tensor([int(p) for p in parts[1:]])
                
                ground_truth.append({
                    "timestamp": timestamp,
                    "annot": bbox
                })
        return ground_truth