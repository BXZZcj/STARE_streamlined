import os
import sys
from pathlib import Path
import time

import torch
import numpy as np

from dv import AedatFile

rvt_project_root = Path("/mnt/f1590153-780c-408d-b394-7b3b56082548/ESOT500/RVT/RVT")
if rvt_project_root.exists() and str(rvt_project_root) not in sys.path:
    print(f"INFO: Adding RVT project root to sys.path: {rvt_project_root}")
    sys.path.insert(0, str(rvt_project_root))

from data.utils.representations import MixedDensityEventStack

# ======================================================================== #
# == 用户配置区 ==
# ======================================================================== #

# 1. 指定你的 aedat4 文件路径
AEDAT4_FILE_PATH = Path('/mnt/f1590153-780c-408d-b394-7b3b56082548/ESOT500/aedat4/car16.aedat4')

# 2. 定义事件表示的参数 (与 rvt-t 模型在 Gen1 上的配置保持一致)
TIME_WINDOW_US = 50 * 1000  # 每个事件表示(帧)的时间跨度，单位：微秒 (50ms)
NUM_TIME_BINS = 10         # 在一个时间窗口内，再细分成多少个时间箱 (即论文中的 T)

# ======================================================================== #

def main():
    """
    主函数，用最简单直接的方式处理单个 aedat4 文件。
    """
    if not AEDAT4_FILE_PATH.exists():
        raise FileNotFoundError(f"错误: 找不到指定的 aedat4 文件: {AEDAT4_FILE_PATH}")

    print("INFO: --------------------------------")
    print(f"INFO: 正在处理文件: {AEDAT4_FILE_PATH}")
    print(f"INFO: 时间窗口大小: {TIME_WINDOW_US / 1000} ms")
    print(f"INFO: 每个窗口的时间箱数量 (T): {NUM_TIME_BINS}")
    print("INFO: --------------------------------\n")

    # --- 1. 用简单直接的方式读取所有事件 ---
    print("INFO: 正在读取所有事件到内存中...")
    start_time = time.time()
    
    with AedatFile(str(AEDAT4_FILE_PATH)) as f:
        # 获取分辨率
        height, width = f['events'].size
        print(f"INFO: 事件流分辨率 (H x W): {height} x {width}")
        
        # 将所有事件包合并成一个大的 NumPy 数组
        # events 的列是: timestamp, x, y, polarity
        events = np.hstack([packet for packet in f['events'].numpy()])

    if events.size == 0:
        print("错误: aedat4 文件中不包含任何事件。")
        return

    # 将时间戳归一化，以第一个事件的时间为 0
    events['timestamp'] = events['timestamp'] - events['timestamp'][0]
    
    end_time = time.time()
    print(f"INFO: 读取了 {len(events)} 个事件，耗时: {end_time - start_time:.2f} 秒")

    # --- 2. 初始化事件表示生成器 ---
    event_representation_generator = MixedDensityEventStack(
        bins=NUM_TIME_BINS, height=height, width=width)
    
    # --- 3. 【核心】用一个简单的循环按固定时间窗口切割 ---
    event_representations = []
    
    # 获取整个事件流的起止时间
    first_timestamp_us = events['timestamp'][0]
    last_timestamp_us = events['timestamp'][-1]
    
    print("\nINFO: 开始用简单的循环按固定时间窗口切割事件流...")
    current_time_us = first_timestamp_us
    frame_count = 0

    while current_time_us < last_timestamp_us:
        # 定义当前时间窗口的起止时间
        window_start_us = current_time_us
        window_end_us = current_time_us + TIME_WINDOW_US

        # 【直接逻辑】使用 NumPy 的布尔索引来选择当前窗口的事件
        # 这比 Slicer 更直观
        time_mask = (events['timestamp'] >= window_start_us) & (events['timestamp'] < window_end_us)
        window_events = events[time_mask]

        if len(window_events) == 0:
            print(f"  - 跳过空事件窗口 (时间: {window_start_us / 1e6:.3f}s - {window_end_us / 1e6:.3f}s)")
            current_time_us = window_end_us
            continue
        
        frame_count += 1

        # 将 NumPy 数组转换为 PyTorch 张量
        x = torch.from_numpy(window_events['x'].astype(np.int64))
        y = torch.from_numpy(window_events['y'].astype(np.int64))
        p = torch.from_numpy(window_events['polarity'].astype(np.int64))
        t = torch.from_numpy(window_events['timestamp'].astype(np.int64))

        # 调用 RVT 的核心函数来构建事件表示
        ev_repr = event_representation_generator.construct(x=x, y=y, pol=p, time=t)
        
        # 展平 T 和 polarity 维度
        ev_repr_flat = ev_repr.flatten(start_dim=0, end_dim=1)
        
        event_representations.append(ev_repr_flat)
        print(f"  - 已生成第 {frame_count} 帧 (时间: {window_start_us / 1e6:.3f}s - {window_end_us / 1e6:.3f}s), "
              f"包含 {len(window_events)} 个事件。")
        
        # 移动到下一个时间窗口
        current_time_us = window_end_us

    print("\nINFO: 事件流处理完成！")

    # --- 4. 检查输出 ---
    print("\n------ 输出检查 ------")
    if event_representations:
        num_frames = len(event_representations)
        first_frame_shape = event_representations[0].shape
        print(f"成功生成了 {num_frames} 个事件表示 (帧)。")
        print(f"每个表示的形状为 (C, H, W): {first_frame_shape}")
        print(f"其中 C = 2 * T = 2 * {NUM_TIME_BINS} = {first_frame_shape[0]}")
    else:
        print("警告: 未生成任何事件表示。可能是文件过短或没有事件。")
    print("--------------------")

if __name__ == "__main__":
    main()