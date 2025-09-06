import os
import numpy as np
from dv import AedatFile
from typing import List, Dict
import torch

from .base_dataset import BaseDataset

class ESOT500DatasetTraditional20FPS20MS(BaseDataset):
    SAMPLING_WINDOW_MS = 20
    FPS = 20
    FRAME_INTERVAL_MS = 1000/FPS
    
    def __init__(self, dataset_path:str):
        super().__init__(dataset_path)
        
        with open(os.path.join(dataset_path, 'test.txt'), 'r') as f:
            self.sequence_list = f.read().splitlines()

    def get_sequence_list(self):
        return self.sequence_list
    
    def _get_sequence_events(self, sequence_name)->np.ndarray:
        if self.cur_loaded_seq_events_name == sequence_name:
            return self.current_loaded_seq_events
        else:
            self.cur_loaded_seq_events_name = sequence_name
        
        aedat4_path = os.path.join(self.dataset_path, "aedat4", f'{sequence_name}.aedat4')
        
        with AedatFile(aedat4_path) as f:
            self.current_loaded_seq_events = np.hstack([packet for packet in f['events'].numpy()])
            # self.events_start_ts_hs = self.events['timestamp'][0]
            # self.events['timestamp'] = self.events['timestamp'] - self.events_start_ts_hs
            
        return self.current_loaded_seq_events

    def _get_sequence_rgb(self, sequence_name)->np.ndarray:
        if self.cur_loaded_seq_rgb_name == sequence_name:
            return self.current_loaded_seq_frames
        else:
            self.cur_loaded_seq_rgb_name = sequence_name
        
        aedat4_path = os.path.join(self.dataset_path, "aedat4", f'{sequence_name}.aedat4')
        
        with AedatFile(aedat4_path) as f:
            self.current_loaded_seq_frames = []
            for frame in f["frames"]:
                self.current_loaded_seq_frames.append([frame.timestamp, frame.image])
            
        return self.current_loaded_seq_frames

    
    def get_events_duration_sec(self, sequence_name:str)->float:
        events = self._get_sequence_events(sequence_name)
        return (events[-1]['timestamp'] - events[0]['timestamp'])/1e6

    def _get_events_by_regular_duration(self, sequence_name:str, ts_start_sec:float, ts_end_sec:float)->np.ndarray:
        events = self._get_sequence_events(sequence_name)
        evs_start_ts_sec = events[0]["timestamp"]/1e6
        actual_ts_start_sec = ts_start_sec+evs_start_ts_sec
        actual_ts_end_sec = ts_end_sec+evs_start_ts_sec
        return events[(events['timestamp']/1e6 >= actual_ts_start_sec) & (events['timestamp']/1e6 <= actual_ts_end_sec)]

    def get_step_input_by_regular_duration(self, sequence_name:str, ts_start_sec:float, ts_end_sec:float)->np.ndarray:
        return self._get_events_by_regular_duration(sequence_name, ts_start_sec, ts_end_sec)
    
    def get_init_input_by_regular_duration(self, sequence_name:str, ts_start_sec:float, ts_end_sec:float)->np.ndarray:
        return self._get_events_by_regular_duration(sequence_name, ts_start_sec, ts_end_sec)

    def get_earlist_aval_init_input_exact_ts_sec(self, sequence_name:str, init_sampling_window_ms:float)->float:
        gt = self.get_sequence_gt(sequence_name)
        for gt_item in gt:
            if gt_item["timestamp"] >= init_sampling_window_ms/1000.0:
                return gt_item["timestamp"]
        return None
        
    def get_sequence_gt(self, sequence_name:str)->List[Dict]:
        gt_path = os.path.join(self.dataset_path, "annots", f'{sequence_name}.txt')
        ground_truth = []
        with open(gt_path, 'r') as f:
            gt_str_lines = f.readlines()

            for gt_str in gt_str_lines:
                gt_ts_sec = float(gt_str.split()[0])
                if gt_ts_sec >= self.SAMPLING_WINDOW_MS/1000:
                    init_ts_ms = int(gt_ts_sec*1e3)
                    init_window_left_ms = init_ts_ms - self.FRAME_INTERVAL_MS
                    break
                else:
                    continue

            for gt_str in gt_str_lines[1:]:
                parts = gt_str.split()
                timestamp = float(parts[0])
                bbox = torch.tensor([int(p) for p in parts[1:]])
                
                if int(timestamp*1e3 - init_window_left_ms)%self.FRAME_INTERVAL_MS != 0:
                    continue
                else:
                    ground_truth.append({
                        "timestamp": timestamp+self.FRAME_INTERVAL_MS/1000+1e-5,
                        "annot": bbox
                    })
        return ground_truth