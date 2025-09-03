import os
import sys
import argparse

prj_path = os.path.join("/home/zongyouyu/nc/STARE_streamlined/libs/EventVOT_Benchmark/HDETrack")
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation.tracker import Tracker

hdetrack_scheduler = Tracker(name="hdetrack", parameter_name="hdetrack_eventvot", dataset_name=None, run_id=None)

hdetrack_params = hdetrack_scheduler.get_parameters()
hdetrack_params.debug = 0
hdetrack_tracker = hdetrack_scheduler.create_tracker(hdetrack_params)

airplane2_20_w2ms_00000 = hdetrack_scheduler._read_image("/home/zongyouyu/nc/ESOT500/20_w2ms/airplane2/VoxelGridComplex/00000.jpg")

init_info = {'init_bbox': [0, 0, 100, 100]}
init_out = hdetrack_tracker.initialize(image=airplane2_20_w2ms_00000, start_frame_idx=0, info=init_info)
print(init_out)

airplane2_20_w2ms_00001 = hdetrack_scheduler._read_image("/home/zongyouyu/nc/ESOT500/20_w2ms/airplane2/VoxelGridComplex/00001.jpg")
track_out = hdetrack_tracker.track(image=airplane2_20_w2ms_00001, info=None)

print(track_out)