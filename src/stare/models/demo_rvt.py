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


class DemoRVT(BasePerceptionModel):
    def __init__(self, config: dict, evs_height: int, evs_width: int, checkpoint_path: str):
        super().__init__(config, evs_height, evs_width, checkpoint_path)
        
        self.rvt_project_path = Path(config["model"]['rvt_project_root'])
        rvt_config_path = self.rvt_project_path / 'config'
        rvt_checkpoint_path = Path(checkpoint_path)
        
        # --- Dynamically add RVT project path ---
        if self.rvt_project_path.exists() and str(self.rvt_project_path) not in sys.path:
            print(f"INFO: Adding RVT project root to sys.path: {self.rvt_project_path}")
            sys.path.insert(0, str(self.rvt_project_path))
        else:
            print("INFO: RVT project root is already in sys.path or does not exist.")
            
        # delay import, ensure path is added
        from config.modifier import dynamically_modify_train_config
        from modules.utils.fetch import fetch_model_module
        
        self.event_repr_config = config["model"]["representation"]
        self.evs_height = evs_height
        self.evs_width = evs_width
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # --- Load RVT config ---
        overrides = [
            'dataset=gen1',
            f'checkpoint={str(rvt_checkpoint_path)}',
            '+experiment/gen1=tiny',
        ]
        
        print("INFO: Loading RVT config...")
        with hydra.initialize_config_dir(config_dir=str(rvt_config_path.absolute()), version_base='1.2'):
            self.rvt_config = hydra.compose(config_name='val', overrides=overrides)

        dynamically_modify_train_config(self.rvt_config)
        
        # store the expected input resolution of the model
        self.final_model_resolution = tuple(self.rvt_config.model.backbone.in_res_hw)

        # --- Load pre-trained model ---
        print("INFO: Loading RVT model...")
        model_class = fetch_model_module(config=self.rvt_config)
        self.model = model_class.load_from_checkpoint(str(rvt_checkpoint_path), **{'full_config': self.rvt_config})
        
        self.model.eval()
        self.model.to(device)
        self.device = device
        print(f"INFO: Model successfully loaded to {self.device}")
        
        self.hidden_state = None
        print("INFO: DemoRVT initialized successfully.")
        
    def _scale_input_resolution(self, evs_repr: torch.Tensor) -> torch.Tensor:
        """Scales the input event representation to the model's desired resolution."""
        # The input tensor is expected to be on the correct device already
        return F.interpolate(
            evs_repr, 
            size=self.final_model_resolution, 
            mode='bilinear', 
            align_corners=False
        )
    
    def _descale_output_resolution(self, scaled_results: torch.Tensor) -> torch.Tensor:
        """
        Descales the bounding box coordinates from the model's input resolution
        back to the original event sensor resolution. Also converts format
        from xyxy to xywh.
        """
        if scaled_results.shape[0] == 0:
            return scaled_results

        H_orig, W_orig = self.evs_height, self.evs_width
        H_scaled, W_scaled = self.final_model_resolution

        scale_w = W_orig / W_scaled
        scale_h = H_orig / H_scaled

        final_results_orig_scale = scaled_results.clone()

        # Apply scaling to xyxy coordinates
        final_results_orig_scale[:, 0] *= scale_w # x1
        final_results_orig_scale[:, 1] *= scale_h # y1
        final_results_orig_scale[:, 2] *= scale_w # x2
        final_results_orig_scale[:, 3] *= scale_h # y2

        # Convert coordinates from xyxy to xywh
        final_results_orig_scale[:, 2] = final_results_orig_scale[:, 2] - final_results_orig_scale[:, 0] # width = x2 - x1
        final_results_orig_scale[:, 3] = final_results_orig_scale[:, 3] - final_results_orig_scale[:, 1] # height = y2 - y1

        return final_results_orig_scale
        
    def initialize(self, events: np.ndarray, info: Dict) -> None:
        """Resets the recurrent state of the model."""
        print("INFO: Initializing/Resetting RVT hidden state.")
        self.hidden_state = None
        # Initialization does not need to return a value.

    def predict(self, events: np.ndarray, info: Dict) -> torch.Tensor:
        """
        Performs a prediction step on a new chunk of events.
        """
        # Dynamically import postprocess function as it's part of the RVT project
        from models.detection.yolox.utils.boxes import postprocess
            
        # --- Stage 1: Pre-processing ---
        # Convert raw events to representation tensor
        evs_repr = self._get_representation(events)
        # Add batch dimension and move to device
        evs_repr = evs_repr.unsqueeze(0).to(self.device)
        # Scale to model's input resolution
        scaled_evs_repr = self._scale_input_resolution(evs_repr)
    
        with torch.inference_mode():
            # --- Stage 2: Model Inference ---
            predictions, _, self.hidden_state = self.model(scaled_evs_repr, self.hidden_state)
            
            # --- Stage 3: Post-processing (Decoding and NMS) ---
            pred_processed = postprocess(
                prediction=predictions,
                num_classes=self.rvt_config.model.head.num_classes,
                conf_thre=self.rvt_config.model.postprocess.confidence_threshold,
                nms_thre=self.rvt_config.model.postprocess.nms_threshold
            )
            final_dets = pred_processed[0]

            # --- Stage 4: Formatting the results ---
            if final_dets is not None and final_dets.shape[0] > 0:
                processed_results = []
                for det in final_dets:
                    box = det[:4] # xyxy
                    final_confidence = det[4] * det[5] # obj_conf * class_conf
                    class_idx = det[6]
                    result_row = torch.cat([box, final_confidence.unsqueeze(0), class_idx.unsqueeze(0)])
                    processed_results.append(result_row)
                scaled_results = torch.stack(processed_results) # Shape (D, 6), format [x1, y1, x2, y2, conf, class_id]
            else:
                # Ensure device is correct for empty tensor
                scaled_results = torch.empty((0, 6), dtype=torch.float32, device=self.device)

            # --- Stage 5: Coordinate Descaling ---
            final_results = self._descale_output_resolution(scaled_results)
            
            # Return results on CPU as is standard for framework integration
            return final_results.to("cpu")
    
    def _get_representation(self, events: np.ndarray) -> torch.Tensor:
        """Converts raw events to a tensor representation."""
        return events_to_representation(
            events, 
            self.event_repr_config, 
            (self.evs_height, self.evs_width)
        )