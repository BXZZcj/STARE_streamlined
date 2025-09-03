import time
from typing import Dict, Callable, List, Tuple, Optional
import numpy as np

from stare.models.base_model import BasePerceptionModel
from stare.utils.convert_event_img import *
from stare.core.metrics import calculate_metric
from stare.data import BaseDataset


class STARE:
    def __init__(self, model: BasePerceptionModel, dataset: BaseDataset, config):
        self.model = model
        self.dataset = dataset
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
        
        
    # def _get_events_by_duration(self, evs:np.ndarray, ts_start_sec:float, ts_end_sec:float)->np.ndarray:
    #     return evs[(evs['timestamp']/1e6 >= ts_start_sec) & (evs['timestamp']/1e6 <= ts_end_sec)]
    
    
    # def _get_events_duration_sec(self, evs:np.ndarray)->float:
    #     return (evs[-1]['timestamp'] - evs[0]['timestamp'])/1e6
    
    
    def _find_gt_for_init(self, ground_truth:List[Dict], init_sampling_window_ms:float)->Tuple:
        init_sampling_window_sec = init_sampling_window_ms/1000.0
        for gt in ground_truth:
            if gt["timestamp"] >= init_sampling_window_sec:
                return gt["timestamp"], gt["annot"]
        return None, None


    def run(self, sequence_name:str, ground_truth:List[Dict])->List[Dict]:
        """Main evaluation loop of STARE (Continuous Sampling & Time Advancement)"""
        
        world_time_sec = self.config['sampling_window_ms']/1000.0
        last_output = None
        evs_duration_sec = self.dataset.get_events_duration_sec(sequence_name)
        outputs_with_timestamps = []
        
        # 1. Initialization
        init_start_time = time.perf_counter()
        if self.config["initialization"]["enabled"]:
            init_sampling_window_ms = self.config["initialization"]['sampling_window_ms']
            if self.config["initialization"]["initialize_timestamp"] == "same_as_window":
                initialize_timestamp_ms = init_sampling_window_ms
            elif self.config["initialization"]["initialize_timestamp"] == "earlist_rgb_frame":
                initialize_timestamp_ms = self.dataset.get_earlist_aval_rgb_regular_ts_sec(sequence_name, init_sampling_window_ms)*1e3
            else:
                raise ValueError(f"Unknown initialize_timestamp mode: {self.config['initialization']['initialize_timestamp']}")

            initial_input = self.dataset.get_init_input_by_regular_duration(sequence_name = sequence_name, ts_start_sec=initialize_timestamp_ms/1e3-init_sampling_window_ms/1e3, ts_end_sec=initialize_timestamp_ms/1e3)
            gt_ts_for_init, gt_annot_for_init = self._find_gt_for_init(ground_truth, initialize_timestamp_ms)
            info_for_init = dict(self.config["initialization"])
            info_for_init["gt_ts_for_init"] = gt_ts_for_init
            info_for_init["gt_annot_for_init"] = gt_annot_for_init.tolist()
            initial_output = self.model.initialize(init_input = initial_input, info = info_for_init)
            
            if initial_output is not None:
                outputs_with_timestamps.append({
                    'output': initial_output,
                    'timestamp': gt_ts_for_init
                })
        
            world_time_sec = gt_ts_for_init
            last_output = gt_annot_for_init
        
        init_end_time = time.perf_counter()
        real_init_latency = init_end_time - init_start_time
        simulated_init_latency = self._get_sim_init_latency(real_init_latency)
        world_time_sec += simulated_init_latency
        
        # 2. Continuous sampling and evaluation loop
        while world_time_sec < evs_duration_sec:
            predict_start_time = time.perf_counter()
            # (1) Sampling: get data from the event stream based on current world time and window size
            ts_start_sec = world_time_sec-self.config['sampling_window_ms']/1000.0
            ts_end_sec = world_time_sec
            step_input = self.dataset.get_step_input_by_regular_duration(sequence_name, ts_start_sec, ts_end_sec)

            # (2) Inference: perform one prediction and return real inference time
            output = self.model.predict(step_input, info={"last_output": last_output, "input_ts_sec": ts_end_sec})
            predict_end_time = time.perf_counter()
            real_predict_latency = predict_end_time - predict_start_time
            
            # (3) Latency simulation: calculate latency for this cycle using hardware simulator
            simulated_predict_latency = self._get_sim_predict_latency(real_predict_latency)
            
            # (4) Advance time: advance the world clock forward
            world_time_sec += simulated_predict_latency
            
            # (5) Record results: store prediction results with timestamps
            outputs_with_timestamps.append({
                'output': output,
                'timestamp': world_time_sec 
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
                if self.config["initialization"]["enabled"]:
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
        eval_results:List = calculate_metric(matched_pairs, metric_configs=self.config["metrics"])
        
        return eval_results