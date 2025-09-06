from dv import AedatFile
import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import torch.nn.functional as F
from torch.nn.init import xavier_normal_ as x_init
import itertools
import visdom
from torch.utils import data
from torchvision.utils import make_grid
import numpy as np
import time
from torch.utils.data import DataLoader
import skimage.measure as skm
import skimage as ski
import random
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import scipy.io as scio
from typing import List

from .base_model import BasePerceptionModel
from stare.utils.convert_event_repr import *

import os
import sys
import argparse
from types import SimpleNamespace

prj_path = os.path.join("/home/zongyouyu/nc/STARE_streamlined/libs/Event-tracking/Evaluate")
if prj_path not in sys.path:
    sys.path.append(prj_path)


from event_loader_02 import Event_dataset
from TrackNet import ETracking_Net
from utils import load_checkpoint
from models import build_model
from config import get_config
from ConfigFunction import parse_option
from util.box_ops import box_xywh_to_xyxy, box_iou1, box_cxcywh_to_xyxy
from Tracker import Event_tracker



class EGT(BasePerceptionModel):
    """
    This is a DIMP18 model that is used to test the framework.
    """
    def __init__(self, config: dict, evs_height: int, evs_width: int, checkpoint_path=None):
        super().__init__(config, evs_height, evs_width, checkpoint_path=None)
        
        self.event_repr_config = config["model"]["representation"]
        self.evs_height = evs_height
        self.evs_width = evs_width

        _, config = parse_option()
        SwinTrans = build_model(config)
        SwinTrans.cuda()
        ETracKNet = ETracking_Net(SwinT = SwinTrans).cuda()
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        OutDict = state_dict['Dict']
        ETracKNet = self.load_model(ETracKNet, OutDict)
        opt = SimpleNamespace()
        opt.update_inter = 2
        opt.search_size = 0.25
        self.Evtracker = Event_tracker(ETracKNet,opt)
        self.Evtracker.eval()

    def load_model(self, model,mode_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        state_dict_m = model.state_dict()

        for k, v in mode_dict.items():
            name = k[7:] # remove `module.`
            if 'Iou_Net' not in name:
                new_state_dict[name] = v
        
        state_dict_m.update(new_state_dict)
        model.load_state_dict(state_dict_m)
        model.eval()
        return model
    
    def create_template(
        self,
        events: np.ndarray, 
        bbox: tuple,
        num_template_points: int = 10000
    ) -> torch.Tensor:
        num_total_events = events.shape[0]
        if num_total_events == 0:
            return torch.empty((5, 0), dtype=torch.float32)

        x_min, y_min = bbox[0], bbox[1]
        x_max, y_max = bbox[0] + bbox[2], bbox[1] + bbox[3]
        event_x, event_y = events['x'], events['y']
        fg_mask = (event_x >= x_min) & (event_x <= x_max) & (event_y >= y_min) & (event_y <= y_max)
        foreground_events = events[fg_mask]
        num_fg_events = foreground_events.shape[0]
        if num_fg_events == 0:
            return torch.empty((5, 0), dtype=torch.float32)
        replace = num_fg_events < num_template_points
        sample_indices = np.random.choice(num_fg_events, size=num_template_points, replace=replace)
        sampled_fg_events = foreground_events[sample_indices]

        template_4d = np.stack([
            sampled_fg_events['x'], 
            sampled_fg_events['y'], 
            sampled_fg_events['timestamp'], 
            sampled_fg_events['polarity']
        ], axis=1).astype(np.float32)

        indicator_column = np.ones((num_template_points, 1), dtype=np.float32)
        template_5d_np = np.hstack([template_4d, indicator_column])

        template_5d_np[:, 0] = template_5d_np[:, 0] / self.evs_width
        template_5d_np[:, 1] = template_5d_np[:, 1] / self.evs_height
        if template_5d_np.shape[0] > 0:
            template_5d_np[:, 2] = template_5d_np[:, 2] - template_5d_np[0, 2] 
        template_5d_np[:, 2] = template_5d_np[:, 2] / 1e6
        template_5d_np[:, 3] = template_5d_np[:, 3] * 2 - 1
        
        return torch.from_numpy(template_5d_np.transpose())

    def get_centered_xywh_bbox(self, top_left_xywh:List, normalize:bool=False):
        cx = top_left_xywh[0] + top_left_xywh[2] / 2
        cy = top_left_xywh[1] + top_left_xywh[3] / 2
        w = top_left_xywh[2]
        h = top_left_xywh[3]
        
        if normalize:
            cx /= self.evs_width
            cy /= self.evs_height
            w /= self.evs_width
            h /= self.evs_height
            
        return [cx, cy, w, h]

    def get_topleft_xywh_bbox(self, centered_xywh:List, denormalize:bool=False)->List:
        if denormalize:
            centered_xywh[0] *= self.evs_width
            centered_xywh[1] *= self.evs_height
            centered_xywh[2] *= self.evs_width
            centered_xywh[3] *= self.evs_height
        return [centered_xywh[0] - centered_xywh[2]/2, centered_xywh[1] - centered_xywh[3]/2, centered_xywh[2], centered_xywh[3]]
        
    def initialize(self, init_input:np.ndarray, info: Dict)->torch.Tensor:
        egt_init_input = self.create_init_input(init_input, bbox=info["gt_annot_for_init"])

        init_box = torch.tensor(self.get_centered_xywh_bbox(info["gt_annot_for_init"], normalize=True), dtype=torch.float32).cuda()
        self.Evtracker._init_box(init_box)

        _ = self.Evtracker.encoding_template(egt_init_input.unsqueeze(0).cuda())
        return None


    def predict(self, step_input:np.ndarray, info: Dict)->torch.Tensor:
        egt_step_input = self._get_representation(step_input)

        with torch.no_grad():
            step_output_raw = self.Evtracker.eval_data(egt_step_input.unsqueeze(0).cuda(), GT=None)

        step_output = self.get_topleft_xywh_bbox(step_output_raw[0].squeeze().cpu().detach().numpy().tolist(), denormalize=True)

        return torch.tensor(step_output)
    
    def _get_representation(self, events: np.ndarray)->torch.Tensor:
        return events_to_representation(events, self.event_repr_config, (self.evs_height, self.evs_width))