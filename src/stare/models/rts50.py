import os
import sys
import torch
import numpy as np
from typing import List, Dict
from torchvision.transforms import ToPILImage

from .base_model import BasePerceptionModel
from stare.utils.convert_event_repr import *

prj_path = os.path.join("/home/zongyouyu/nc/STARE/lib/pytracking")
if prj_path not in sys.path:
    sys.path.append(prj_path)

from pytracking.evaluation import Tracker


class RTS50(BasePerceptionModel):
    """
    This is a RTS50 model that is used to test the framework.
    """
    def __init__(self, config: dict, evs_height: int, evs_width: int, checkpoint_path=None):
        super().__init__(config, evs_height, evs_width, checkpoint_path=None)
        
        self.event_repr_config = config["model"]["representation"]
        self.evs_height = evs_height
        self.evs_width = evs_width

        self.to_pil_image = ToPILImage()
        
        rts50_scheduler = Tracker(name="rts", parameter_name="rts50", run_id=None)

        rts50_params = rts50_scheduler.get_parameters()
        rts50_params.debug = 0
        self.rts50_tracker = rts50_scheduler.create_tracker(rts50_params)

        self.last_segmentation_raw = None
        
    def initialize(self, init_input:np.ndarray, info: Dict)->torch.Tensor:
        evs_repr = self._get_representation(init_input).cpu()#.numpy().transpose(1, 2, 0)
        evs_repr = np.array(self.to_pil_image(evs_repr))
        info["init_bbox"] = info["gt_annot_for_init"]
        # It will return {"time": XXXX}
        init_output = self.rts50_tracker.initialize(image=evs_repr, info=info) 
        self.last_segmentation_raw = init_output['segmentation_raw']
        return None


    def predict(self, step_input:np.ndarray, info: Dict)->torch.Tensor:
        evs_repr = self._get_representation(step_input).cpu()#.numpy().transpose(1, 2, 0)
        evs_repr = np.array(self.to_pil_image(evs_repr))
        track_out = self.rts50_tracker.track(image=evs_repr, info={'previous_output':{'segmentation_raw':self.last_segmentation_raw}})
        return torch.tensor(track_out["target_bbox"])
    
    def _get_representation(self, events: np.ndarray)->torch.Tensor:
        return events_to_representation(events, self.event_repr_config, (self.evs_height, self.evs_width))