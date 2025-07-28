import numpy as np
import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf, open_dict
import time
from typing import List, Dict

from .base_model import BasePerceptionModel
from stare.utils.convert_event_img import *


class DemoDEVO(BasePerceptionModel):
    def __init__(self, config: dict, evs_height: int, evs_width: int, checkpoint_path: str):
        super().__init__(config, evs_height, evs_width, checkpoint_path)
        
        self.devo_project_path = Path(config['model']['devo_project_root'])
        devo_config_path = self.devo_project_path / 'config/default.yaml'
        devo_checkpoint_path = Path(checkpoint_path)
        
        # --- Dynamically add DEVO project path ---
        if self.devo_project_path.exists() and str(self.devo_project_path) not in sys.path:
            print(f"INFO: Adding DEVO project root to sys.path: {self.devo_project_path}")
            sys.path.insert(0, str(self.devo_project_path))
        else:
            print("INFO: DEVO project root is already in sys.path or does not exist.")
            
        # delay import, ensure path is added
        from devo.devo import DEVO
        from devo.config import cfg
        
        self.event_repr_config = config["model"]["representation"]
        self.evs_height = evs_height
        self.evs_width = evs_width
        
        self.intrinsic = config["dataset"]["intrisic"]
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        config = cfg
        config.merge_from_file(devo_config_path)
        config.MIXED_PRECISION = True
        
        self.slam = DEVO(config, checkpoint_path, evs=True, ht=evs_height, wd=evs_width)
        
    def initialize(self, events: np.ndarray, info: Dict) -> None:
        pass

    def predict(self, events: np.ndarray, info: Dict) -> torch.Tensor:
        evs_repr = self._get_representation(events)
        evs_repr = evs_repr.to(self.device)
        
        tstamp_us = torch.tensor(info["input_ts_sec"] * 1e6, device="cuda")
        intrinsics = torch.tensor(self.intrinsic, device="cuda")
    
        with torch.inference_mode():
            self.slam(tstamp_us, evs_repr, intrinsics)

            if self.slam.n > 0: # ensure at least one frame is processed
                current_pose = self.slam.poses_[self.slam.n - 1].cpu().numpy()
                return torch.tensor(current_pose)
    
    def _get_representation(self, events: np.ndarray) -> torch.Tensor:
        """Converts raw events to a tensor representation."""
        return events_to_representation(
            events, 
            self.event_repr_config, 
            (self.evs_height, self.evs_width)
        )