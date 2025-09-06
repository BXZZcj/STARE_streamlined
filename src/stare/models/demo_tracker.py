import torch
import numpy as np
from typing import List, Dict

from .base_model import BasePerceptionModel
from stare.utils.convert_event_repr import *


class DemoTracker(BasePerceptionModel):
    """
    This is a simple demo tracker that is used to test the framework.
    """
    def __init__(self, config: dict, evs_height: int, evs_width: int, checkpoint_path: str):
        super().__init__(config, evs_height, evs_width, checkpoint_path)
        
        # This class is just a demo, so we don't need to load a model
        checkpoint_path = None
        self.checkpoint_path = checkpoint_path
        self.model = torch.load(checkpoint_path) if checkpoint_path is not None else lambda x, y: torch.tensor([0,0,1,1])
        
        self.event_repr_config = config["model"]["representation"]
        self.evs_height = evs_height
        self.evs_width = evs_width
        
    def initialize(self, init_input:np.ndarray, info: Dict)->torch.Tensor:
        evs_repr = self._get_representation(init_input)
        return torch.tensor(info["gt_annot_for_init"])

    def predict(self, step_input:np.ndarray, info: Dict)->torch.Tensor:
        evs_repr = self._get_representation(step_input)
        return self.model(step_input, info)
    
    def _get_representation(self, events: np.ndarray)->torch.Tensor:
        return events_to_representation(events, self.event_repr_config, (self.evs_height, self.evs_width))