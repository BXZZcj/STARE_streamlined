﻿from abc import ABC, abstractmethod
from typing import Dict
import torch

class BasePerceptionModel(ABC):
    """Abstract base class for all perception models"""
    @abstractmethod
    def __init__(self, config: dict, evs_height: int, evs_width: int, checkpoint_path: str):
        pass
    
    @abstractmethod
    def initialize(self, init_input:torch.Tensor, info:Dict=None)->torch.Tensor:
        """
        Initialize the model with the first frame data (crucial for stateful tasks like VOT).
        For stateless tasks (e.g., detection), this can be a no-op.
        """
        pass

    @abstractmethod
    def predict(self, step_input:torch.Tensor, info:Dict=None)->torch.Tensor:
        """
        Perform one forward inference.

        Returns:
            prediction: model output (e.g., bounding box)
            inference_time_sec: actual inference time for this call (seconds)
        """
        pass