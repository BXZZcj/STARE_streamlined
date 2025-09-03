import os
import sys
import argparse

prj_path = os.path.join("/home/zongyouyu/nc/STARE_streamlined/libs/Mamba_FETrack/Mamba_FETrackV2")
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation.tracker import Tracker

mamba_fetrackV2_scheduler = Tracker(name="mamba_fetrack", parameter_name="mamba_fetrack_felt", dataset_name=None, run_id=None)

mamba_fetrackV2_params = mamba_fetrackV2_scheduler.get_parameters()
mamba_fetrackV2_params.debug = 0
mamba_fetrackV2_tracker = mamba_fetrackV2_scheduler.create_tracker(mamba_fetrackV2_params)

airplane2_20_w2ms_00000_event = mamba_fetrackV2_scheduler._read_image("/home/zongyouyu/nc/ESOT500/20_w2ms/airplane2/VoxelGridComplex/00000.jpg")
airplane2_20_w2ms_00000_rgb = mamba_fetrackV2_scheduler._read_image("/home/zongyouyu/nc/ESOT500/20_w2ms/airplane2/VoxelGridComplex/00000.jpg")

init_info = {'init_bbox': [0, 0, 100, 100]}
init_out = mamba_fetrackV2_tracker.initialize(image=airplane2_20_w2ms_00000_rgb, event_image=airplane2_20_w2ms_00000_event, start_frame_idx=0, info=init_info)
print(init_out)

airplane2_20_w2ms_00001_event = mamba_fetrackV2_scheduler._read_image("/home/zongyouyu/nc/ESOT500/20_w2ms/airplane2/VoxelGridComplex/00001.jpg")
airplane2_20_w2ms_00001_rgb = mamba_fetrackV2_scheduler._read_image("/home/zongyouyu/nc/ESOT500/20_w2ms/airplane2/VoxelGridComplex/00001.jpg")
track_out = mamba_fetrackV2_tracker.track(image=airplane2_20_w2ms_00001_rgb, event_img=airplane2_20_w2ms_00001_event, info=None)

print(track_out)