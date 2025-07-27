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

rvt_project_root = Path("/mnt/f1590153-780c-408d-b394-7b3b56082548/ESOT500/RVT/RVT")
if rvt_project_root.exists() and str(rvt_project_root) not in sys.path:
    print(f"INFO: Adding RVT project root to sys.path: {rvt_project_root}")
    sys.path.insert(0, str(rvt_project_root))

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_model_module
from models.detection.yolox.utils.boxes import postprocess
from utils.padding import InputPadderFromShape

class RVT(BasePerceptionModel):
    """
    This is a simple demo tracker that is used to test the framework.
    """
    def __init__(self, event_repr_config: dict, evs_height: int, evs_width: int, checkpoint_path: str):
        super().__init__(event_repr_config, evs_height, evs_width, checkpoint_path)
        
        self.event_repr_config = event_repr_config
        self.evs_height = evs_height
        self.evs_width = evs_width
        
        rvt_root_path = Path(event_repr_config["params"]['rvt_project_root'])
        rvt_config_path = rvt_root_path / 'config'
        rvt_checkpoint_path = checkpoint_path

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        overrides = [
            'dataset=gen1',
            f'checkpoint={str(rvt_checkpoint_path)}',
            '+experiment/gen1=tiny',
        ]
        
        print("INFO: Loading RVT config...")
        with hydra.initialize_config_dir(config_dir=str(rvt_config_path.absolute()), version_base='1.2'):
            self.rvt_config = hydra.compose(config_name='val', overrides=overrides)

        # esot500_resolution_hw = (evs_height, evs_width)    
        # print(f"\nINFO: Trying to modify resolution to: {esot500_resolution_hw}")

        # with open_dict(self.rvt_config.dataset):
        #     self.rvt_config.dataset.resolution_hw = list(esot500_resolution_hw)

        # print("INFO: Running dynamic config function to adapt to new resolution...")
        dynamically_modify_train_config(self.rvt_config)
        
        self.final_model_resolution = tuple(self.rvt_config.model.backbone.in_res_hw)
        # print(f"INFO: Model successfully adapted! Final effective input resolution is: {final_model_resolution}")
        # self.input_padder = InputPadderFromShape(desired_hw=final_model_resolution)
        
        # ======================================================================== #

        print("INFO: Loading model...")
        model_class = fetch_model_module(config=self.rvt_config)
        self.model = model_class.load_from_checkpoint(str(rvt_checkpoint_path), **{'full_config': self.rvt_config})
        
        self.model.eval()
        self.model.to(device)
        print(f"INFO: Model successfully loaded to {device}")
        
    def _scale_input_resolution(
        self, 
        evs_repr: torch.Tensor # shape = (B, C, H, W)
    )->torch.Tensor:
        evs_repr_scaled = F.interpolate(evs_repr, size = self.final_model_resolution, mode = 'bilinear', align_corners = False)
        return evs_repr_scaled
    
    def _descale_output_resolution(
        self,
        output
    ):
        pass
        
    def initialize(self, events:np.ndarray, info: Dict)->torch.Tensor:
        self.hidden_state = None
        return None

    def predict(self, events:np.ndarray, info: Dict)->torch.Tensor:
        evs_repr = self._get_representation(events).unsqueeze(0)
        scaled_evs_repr = self._scale_input_resolution(evs_repr)
    
        with torch.inference_mode():            
            # invoke the forward function of pl_module, it will proxy to self.mdl.forward
            predictions, _, self.hidden_state = self.model(scaled_evs_repr, self.hidden_state)
            
            # pl_module.mdl.forward returns a list, each element corresponds to a sample in the batch
            # postprocess function also expects a list
            pred_processed = postprocess(
                prediction=predictions,
                num_classes=self.rvt_config.model.head.num_classes,
                conf_thre=self.rvt_config.model.postprocess.confidence_threshold,
                nms_thre=self.rvt_config.model.postprocess.nms_threshold
            )
            # postprocess returns a list, we take the result of the only sample in the batch
            final_dets = pred_processed[0]

            if final_dets is not None and final_dets.shape[0] > 0:
                num_detections = final_dets.shape[0]
                print(f"  Detected {num_detections} objects.")

                processed_results = []
                for det in final_dets:
                    box = det[:4]
                    final_confidence = det[4] * det[5]
                    class_idx = det[6]
                    result_row = torch.cat([
                        box, 
                        final_confidence.unsqueeze(0), 
                        class_idx.unsqueeze(0)
                    ])
                    processed_results.append(result_row)
                
                # The shape of scaled_results is (D, 6), coordinates in scaled space
                scaled_results = torch.stack(processed_results)

            else:
                print("  No objects detected.")
                scaled_results = torch.empty((0, 6), dtype=torch.float32, device=predictions[0].device)

            # If there are no detection results, return an empty tensor
            if scaled_results.shape[0] == 0:
                return scaled_results

            # --- Coordinate reverse scaling ---
            
            # 1. Get the original and scaled sizes
            H_orig, W_orig = self.evs_height, self.evs_width
            H_scaled, W_scaled = self.final_model_resolution

            # 2. Calculate the scaling ratio
            scale_w = W_orig / W_scaled
            scale_h = H_orig / H_scaled

            final_results_orig_scale = scaled_results.clone()

            # 3. Apply scaling
            # scaled_results[:, 0] is x1, scaled_results[:, 1] is y1, scaled_results[:, 2] is x2, scaled_results[:, 3] is y2
            final_results_orig_scale[:, 0] *= scale_w
            final_results_orig_scale[:, 1] *= scale_h
            final_results_orig_scale[:, 2] *= scale_w
            final_results_orig_scale[:, 3] *= scale_h
            final_results_orig_scale[:, 2] = final_results_orig_scale[:, 2] - final_results_orig_scale[:, 0]
            final_results_orig_scale[:, 3] = final_results_orig_scale[:, 3] - final_results_orig_scale[:, 1]

            return final_results_orig_scale.to("cpu") # shape = (number of detections, 6), i.e. (x, y, w, h, confidence, class_idx)
    
    def _get_representation(self, events: np.ndarray)->torch.Tensor:
        return events_to_representation(events, self.event_repr_config, (self.evs_height, self.evs_width))