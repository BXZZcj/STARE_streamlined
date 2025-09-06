import os
import sys
import torch
import numpy as np
from typing import List, Dict

from .base_model import BasePerceptionModel
from stare.utils.convert_event_img import *

prj_path = os.path.join("/home/zongyouyu/nc/STARE_streamlined/libs/EventVOT_Benchmark/HDETrack")
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation.tracker import Tracker


class HDETrack(BasePerceptionModel):
    """
    This is a HDETrack model that is used to test the framework.
    """
    def __init__(self, config: dict, evs_height: int, evs_width: int, checkpoint_path=None):
        super().__init__(config, evs_height, evs_width, checkpoint_path=None)
        
        self.event_repr_config = config["model"]["representation"]
        self.evs_height = evs_height
        self.evs_width = evs_width
        
        hdetrack_scheduler = Tracker(name="hdetrack", parameter_name="hdetrack_eventvot", dataset_name=None, run_id=None)

        hdetrack_params = hdetrack_scheduler.get_parameters()
        hdetrack_params.debug = 0
        self.hdetrack_tracker = hdetrack_scheduler.create_tracker(hdetrack_params)
        
    def initialize(self, init_input:np.ndarray, info: Dict)->torch.Tensor:
        evs_repr = self._get_representation(init_input).cpu().numpy().transpose(1, 2, 0)

        info["init_bbox"] = info["gt_annot_for_init"]
        init_out = self.hdetrack_tracker.initialize(image=evs_repr, start_frame_idx=0, info=info)
        return init_out


    def predict(self, step_input:np.ndarray, info: Dict)->torch.Tensor:
        evs_repr = self._get_representation(step_input).cpu().numpy().transpose(1, 2, 0)
        track_out = self.hdetrack_tracker.track(image=evs_repr, info=None)
        return torch.tensor(track_out["target_bbox"])
    
    def _get_representation(self, events: np.ndarray)->torch.Tensor:
        return events_to_representation(events, self.event_repr_config, (self.evs_height, self.evs_width))