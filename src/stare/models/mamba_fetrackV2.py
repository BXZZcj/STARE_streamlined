import os
import sys
import torch
import numpy as np
from typing import List, Dict

from .base_model import BasePerceptionModel
from stare.utils.convert_event_img import *

prj_path = os.path.join("/home/zongyouyu/nc/STARE_streamlined/libs/Mamba_FETrack/Mamba_FETrackV2")
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation.tracker import Tracker


class Mamba_FETrackV2(BasePerceptionModel):
    """
    This is a HDETrack model that is used to test the framework.
    """
    def __init__(self, config: dict, evs_height: int, evs_width: int, checkpoint_path=None):
        super().__init__(config, evs_height, evs_width, checkpoint_path=None)
        
        self.event_repr_config = config["model"]["representation"]
        self.evs_height = evs_height
        self.evs_width = evs_width
        
        mamba_fetrackV2_scheduler = Tracker(name="mamba_fetrack", parameter_name="mamba_fetrack_felt", dataset_name=None, run_id=None)

        mamba_fetrackV2_params = mamba_fetrackV2_scheduler.get_parameters()
        mamba_fetrackV2_params.debug = 0
        self.mamba_fetrackV2_tracker = mamba_fetrackV2_scheduler.create_tracker(mamba_fetrackV2_params)
        
    def initialize(self, init_input:Tuple[np.ndarray, np.ndarray], info: Dict)->torch.Tensor:
        rgb, events = init_input
        evs_repr = self._get_representation(events).cpu().numpy().transpose(1, 2, 0)

        info["init_bbox"] = info["gt_annot_for_init"]
        init_out = self.mamba_fetrackV2_tracker.initialize(image=rgb, event_image=evs_repr, start_frame_idx=0, info=info)
        return init_out


    def predict(self, step_input:Tuple[np.ndarray, np.ndarray], info: Dict)->torch.Tensor:
        rgb, events = step_input
        evs_repr = self._get_representation(events).cpu().numpy().transpose(1, 2, 0)
        track_out = self.mamba_fetrackV2_tracker.track(image=rgb, event_img=evs_repr, info=None)
        return torch.tensor(track_out["target_bbox"])
    
    def _get_representation(self, events: np.ndarray)->torch.Tensor:
        return events_to_representation(events, self.event_repr_config, (self.evs_height, self.evs_width))