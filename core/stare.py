import time
from typing import Dict, Callable, List, Tuple, Optional
import numpy as np

from models.base_model import BasePerceptionModel
from utils.convert_event_img import *
from core.metrics import calculate_metric


class STARE:
    def __init__(self, model: BasePerceptionModel, config):
        self.model = model
        self.config = config
        self.sim_config = self.config['hardware_simulator']


    def _get_sim_predict_latency(self, real_latency_sec):
        """Hardware simulator logic"""
        mode = self.sim_config['predict']['mode']
        if mode == 'real':
            return real_latency_sec
        elif mode == 'fixed':
            return self.sim_config['predict']['fixed_latency_ms'] / 1000.0
        elif mode == 'multiplier':
            return real_latency_sec * self.sim_config['predict']['latency_multiplier']
        else:
            raise ValueError(f"Unknown simulator mode: {mode}")
        
        
    def _get_sim_init_latency(self, real_latency_sec):
        """Hardware simulator logic"""
        mode = self.sim_config['initialize']['mode']
        if mode == 'real':
            return real_latency_sec
        elif mode == 'fixed':
            return self.sim_config['initialize']['fixed_latency_ms'] / 1000.0
        elif mode == 'multiplier':
            return real_latency_sec * self.sim_config['initialize']['latency_multiplier']
        else:
            raise ValueError(f"Unknown simulator mode: {mode}")
        
        
    def _get_events_by_duration(self, evs:np.ndarray, ts_start_sec:float, ts_end_sec:float)->np.ndarray:
        return evs[(evs['timestamp']/1e6 >= ts_start_sec) & (evs['timestamp']/1e6 <= ts_end_sec)]
    
    
    def _get_events_duration_sec(self, evs:np.ndarray)->float:
        return (evs[-1]['timestamp'] - evs[0]['timestamp'])/1e6
    
    
    def _find_gt_for_init(self, ground_truth:List[Dict], init_sampling_window_ms:float)->Tuple:
        init_sampling_window_sec = init_sampling_window_ms/1000.0
        for gt in ground_truth:
            if gt["timestamp"] >= init_sampling_window_sec:
                return gt["timestamp"], gt["annot"]
        return None, None


    def run(self, event_stream:np.ndarray, ground_truth:List[Dict])->List[Dict]:
        """Main evaluation loop of STARE (Continuous Sampling & Time Advancement)"""
        
        world_time = self.config['sampling_window_ms']
        last_output = None
        evs_duration_sec = self._get_events_duration_sec(event_stream)
        outputs_with_timestamps = []
        
        # 1. Initialization
        init_start_time = time.perf_counter()
        if self.config["initialization"]["enabled"]:
            initial_events = self._get_events_by_duration(event_stream, 0, self.config["initialization"]['sampling_window_ms'])
            gt_ts_for_init, gt_annot_for_init = self._find_gt_for_init(ground_truth, self.config["initialization"]['sampling_window_ms'])
            info_for_init = self.config["initialization"]
            info_for_init["gt_ts_for_init"] = gt_ts_for_init
            info_for_init["gt_annot_for_init"] = gt_annot_for_init
            self.model.initialize(events = initial_events, info = info_for_init)
            
            outputs_with_timestamps.append({
                'output': gt_annot_for_init,
                'timestamp': gt_ts_for_init
            })
        
            world_time = gt_ts_for_init
            last_output = gt_annot_for_init
        
        init_end_time = time.perf_counter()
        real_init_latency = init_end_time - init_start_time
        simulated_init_latency = self._get_sim_init_latency(real_init_latency)
        world_time += simulated_init_latency
        
        # 2. Continuous sampling and evaluation loop
        while world_time < evs_duration_sec:
            predict_start_time = time.perf_counter()
            # (1) Sampling: get data from the event stream based on current world time and window size
            ts_start_sec = world_time-self.config['sampling_window_ms']/1000.0
            ts_end_sec = world_time
            event_segment = self._get_events_by_duration(evs=event_stream, ts_start_sec=ts_start_sec, ts_end_sec=ts_end_sec)

            # (2) Inference: perform one prediction and return real inference time
            output:List = self.model.predict(event_segment, info={"last_output": last_output}).tolist()
            predict_end_time = time.perf_counter()
            real_predict_latency = predict_end_time - predict_start_time
            
            # (3) Latency simulation: calculate latency for this cycle using hardware simulator
            simulated_predict_latency = self._get_sim_predict_latency(real_predict_latency)
            
            # (4) Advance time: advance the world clock forward
            world_time += simulated_predict_latency
            
            # (5) Record results: store prediction results with timestamps
            outputs_with_timestamps.append({
                'output': output,
                'timestamp': world_time 
            })
            
            last_output = output  # Update state

        return outputs_with_timestamps

    def evaluate(
        self, 
        outputs_with_timestamps:List[Dict],
        ground_truth:List[Dict]
    )->List[float]:
        """
        Implement the matching logic from the paper to calculate latency-aware metrics,
        optimized by using a single pass through sorted predictions and ground truths.
        Assumes both predictions and ground_truths are sorted by timestamp in ascending order.
        """
        matched_pairs = []
        
        output_idx = 0
        num_outputs = len(outputs_with_timestamps)
        
        # Iterate over each ground truth in ascending order of timestamp
        for gt in ground_truth:
            gt_ts, gt_annot = gt["timestamp"], gt["annot"]
            # If the first output timestamp is greater than the current ground truth timestamp,
            # we need to use the default output or the initial ground truth
            if outputs_with_timestamps[output_idx]["timestamp"] > gt_ts:
                if self.config["evaluation"]["enabled"]:
                    if self.config["evaluation"]["default_output"] == "initial_gt":
                        matched_pairs.append((ground_truth[0]["annot"], gt_annot))
                    else:
                        matched_pairs.append((self.config["evaluation"]["default_output"], gt_annot))
                else:
                    pass
                continue
            
            # Move output_idx forward while next output timestamp <= gt_timestamp
            while output_idx + 1 < num_outputs and outputs_with_timestamps[output_idx + 1]["timestamp"] <= gt_ts:
                output_idx += 1
            
            # Check if current output timestamp is <= gt_timestamp
            if output_idx < num_outputs and outputs_with_timestamps[output_idx]["timestamp"] <= gt_ts:
                matched_pairs.append((outputs_with_timestamps[output_idx]["output"], gt_annot))
            else:
                # No output available before or at this gt_timestamp, skip
                pass

        # Compute metrics on the matched pairs
        eval_results:List[float] = calculate_metric(matched_pairs, metric_configs=self.config["metrics"])
        
        return eval_results