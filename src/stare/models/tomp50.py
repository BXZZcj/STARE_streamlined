import os
import sys
import torch
import numpy as np
from typing import List, Dict

from .base_model import BasePerceptionModel
from stare.utils.convert_event_repr import *
from torchvision.transforms import ToPILImage

prj_path = os.path.join("/home/zongyouyu/nc/STARE/lib/pytracking")
if prj_path not in sys.path:
    sys.path.append(prj_path)

from pytracking.evaluation import Tracker


class TOMP50(BasePerceptionModel):
    """
    This is a TOMP50 model that is used to test the framework.
    """
    def __init__(self, config: dict, evs_height: int, evs_width: int, checkpoint_path=None):
        super().__init__(config, evs_height, evs_width, checkpoint_path=None)
        
        self.event_repr_config = config["model"]["representation"]
        self.evs_height = evs_height
        self.evs_width = evs_width

        self.to_pil_image = ToPILImage()
        
        tomp50_scheduler = Tracker(name="tomp", parameter_name="tomp50", run_id=None)

        tomp50_params = tomp50_scheduler.get_parameters()
        tomp50_params.debug = 0
        self.tomp50_tracker = tomp50_scheduler.create_tracker(tomp50_params)
        
    def initialize(self, init_input:np.ndarray, info: Dict)->torch.Tensor:
        evs_repr = self._get_representation(init_input).cpu()#.numpy().transpose(1, 2, 0)
        evs_repr = np.array(self.to_pil_image(evs_repr))
        info["init_bbox"] = info["gt_annot_for_init"]
        # It will return {"time": XXXX}
        _ = self.tomp50_tracker.initialize(image=evs_repr, info=info) 
        return None


    def predict(self, step_input:np.ndarray, info: Dict)->torch.Tensor:
        evs_repr = self._get_representation(step_input).cpu()#.numpy().transpose(1, 2, 0)
        evs_repr = np.array(self.to_pil_image(evs_repr))
        track_out = self.tomp50_tracker.track(image=evs_repr, info=None)
        return torch.tensor(track_out["target_bbox"])
    
    def _get_representation(self, events: np.ndarray)->torch.Tensor:
        return events_to_representation(events, self.event_repr_config, (self.evs_height, self.evs_width))