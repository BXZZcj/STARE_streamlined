from typing import Dict, Any, Tuple
import numpy as np
import torch
import sys
from pathlib import Path

from .event_utils import events_to_neg_pos_voxel_torch, events_to_voxel_torch


def _prepare_event_tensors(events: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """
    Helper function: convert a NumPy structured event array to PyTorch tensors on GPU.
    This is a commonly reused preparation step.
    """
    xs = torch.from_numpy(events['x'].copy().astype(np.int32)).to(device)
    ys = torch.from_numpy(events['y'].copy().astype(np.int32)).to(device)
    ts = torch.from_numpy(events['timestamp'].copy()).to(device)
    ps = torch.from_numpy(events['polarity'].copy()).to(device)
    return xs, ys, ts, ps

def create_voxel_grid(
    events: np.ndarray,
    resolution: Tuple[int, int],
    device: torch.device,
    num_bins: int = 5,
    temporal_bilinear: bool = True
) -> torch.Tensor:
    """
    Create a two-channel (positive/negative) Voxel Grid from events.
    Returns: a tensor of shape (2 * num_bins, H, W)
    """
    if len(events) == 0:
        return torch.zeros((2 * num_bins, *resolution), device=device)
        
    xs, ys, ts, ps = _prepare_event_tensors(events, device)
    
    voxel_pos, voxel_neg = events_to_neg_pos_voxel_torch(
        xs, ys, ts, ps, num_bins, device, resolution, temporal_bilinear
    )
    return torch.cat((voxel_pos, voxel_neg), dim=0)

def create_complex_voxel_grid(
    events: np.ndarray,
    resolution: Tuple[int, int],
    device: torch.device,
    num_bins: int = 5,
    temporal_bilinear: bool = True
) -> torch.Tensor:
    """
    Create a single-channel Voxel Grid with polarity encoded as +1/-1 from events.
    Returns: a tensor of shape (num_bins, H, W)
    """
    if len(events) == 0:
        return torch.zeros((num_bins, *resolution), device=device)

    xs, ys, ts, ps = _prepare_event_tensors(events, device)
    ps = ps.float() * 2 - 1  # Map {0,1} to {-1,1}
    
    return events_to_voxel_torch(
        xs, ys, ts, ps, num_bins, device, resolution, temporal_bilinear
    )

def create_visual_frame(events: np.ndarray, resolution: Tuple[int, int], **kwargs) -> torch.Tensor:
    """
    Create a visualization event frame (red/blue).
    Returns: a tensor of shape (H, W, 3) on CPU.
    """
    height, width = resolution
    frame = torch.full((height, width, 3), 128, dtype=torch.uint8) # Mid-gray background
    if len(events) > 0:
        y_coords = events['y']
        x_coords = events['x']
        polarities = events['polarity']
        
        pos_mask = polarities == 1
        neg_mask = polarities == 0
        
        # Blue represents negative polarity events
        frame[y_coords[neg_mask], x_coords[neg_mask]] = torch.tensor([255, 0, 0], dtype=torch.uint8)
        # Red represents positive polarity events
        frame[y_coords[pos_mask], x_coords[pos_mask]] = torch.tensor([0, 0, 255], dtype=torch.uint8)
    return frame


def create_stacked_histogram_from_rvt(
    events: np.ndarray, 
    resolution: Tuple[int, int],
    device: torch.device, 
    num_time_bins: int, 
    rvt_project_root: str
) -> torch.Tensor:
    rvt_project_root = Path(rvt_project_root)
    if rvt_project_root.exists() and str(rvt_project_root) not in sys.path:
        print(f"INFO: Adding RVT project root to sys.path: {rvt_project_root}")
        sys.path.insert(0, str(rvt_project_root))
    from data.utils.representations import StackedHistogram
    
    height, width = resolution    
    event_representation_generator = StackedHistogram(
        bins=num_time_bins, 
        height=height, 
        width=width,
        fastmode=True
    )
    
    x = torch.from_numpy(np.copy(events['x']).astype(np.int64))
    y = torch.from_numpy(np.copy(events['y']).astype(np.int64))
    p = torch.from_numpy(np.copy(events['polarity']).astype(np.int64))
    t = torch.from_numpy(np.copy(events['timestamp']).astype(np.int64))

    ev_repr_flat = event_representation_generator.construct(x=x, y=y, pol=p, time=t)
    ev_repr_flat = ev_repr_flat.to(device).to(torch.float32)
    
    return ev_repr_flat

def create_voxel_grid_from_devo(
    events: np.ndarray, 
    resolution: Tuple[int, int],
    device: torch.device,
    num_bins: int = 5,
    remapping_maps: np.ndarray = None
) -> torch.Tensor:
    """
    the version of `to_voxel_grid` (from DEVO source code) that directly receives (N, 4) event array.
    this version keeps the original code's simple style and implementation logic.

    Args:
        events (np.ndarray): the input (N, 4) event slice (x, y, t, p).
        H, W (int): the height and width of the voxel grid.
        nb_of_time_bins (int): the number of time bins (channels).
        remapping_maps (np.ndarray, optional): the remapping maps.

    Returns:
        torch.Tensor: the voxel grid of shape (num_bins, H, W).
    """
    # if the event stream is empty, return an empty grid
    if events.shape[0] == 0:
        return torch.zeros(num_bins, *resolution, dtype=torch.float32)

    xs = events[:]['x']
    ys = events[:]['y']
    ts = events[:]['timestamp']
    ps = events[:]['polarity']

    voxel_grid = torch.zeros(num_bins, *resolution, dtype=torch.float32)
    voxel_grid_flat = voxel_grid.flatten()
    
    ps = ps.astype(np.int8)
    ps[ps == 0] = -1

    # Convert timestamps to [0, nb_of_time_bins] range.
    duration = ts[-1] - ts[0] + 1e-6
    start_timestamp = ts[0]
    
    # note: the original code has a little bit of inefficiency (stack then unpack), but we keep it for the sake of original code.
    features = torch.from_numpy(np.stack([xs.astype(np.float32), ys.astype(np.float32), ts, ps], axis=1))
    x = features[:, 0]
    y = features[:, 1]
    polarity = features[:, 3].float()
    t = (features[:, 2] - start_timestamp) * (num_bins - 1) / duration
    t = t.to(torch.float64)

    if remapping_maps is not None:
        remapping_maps = torch.from_numpy(remapping_maps)
        x_idx = x.round().long().clamp(0, resolution[1]-1)
        y_idx = y.round().long().clamp(0, resolution[0]-1)
        remapped = remapping_maps[:, y_idx, x_idx]
        x, y = remapped[0], remapped[1]

    left_t, right_t = t.floor(), t.floor()+1
    left_x, right_x = x.floor(), x.floor()+1
    left_y, right_y = y.floor(), y.floor()+1

    for lim_x in [left_x, right_x]:
        for lim_y in [left_y, right_y]:
            for lim_t in [left_t, right_t]:
                mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x < resolution[1]) \
                       & (lim_y < resolution[0]) & (lim_t < num_bins)

                if not torch.any(mask): continue
                
                lin_idx = lim_x[mask].long() + lim_y[mask].long() * resolution[1] + lim_t[mask].long() * resolution[1] * resolution[0]
                weight = polarity[mask] * (1-(lim_x[mask]-x[mask]).abs()) * (1-(lim_y[mask]-y[mask]).abs()) * (1-(lim_t[mask]-t[mask]).abs())
                voxel_grid_flat.index_add_(dim=0, index=lin_idx, source=weight.float())

    return voxel_grid


def create_egt_raw_events(
    events: np.ndarray, 
    resolution: Tuple[int, int], 
    device: torch.device
) -> torch.Tensor:
    if not isinstance(events, np.ndarray) or events.shape[0] == 0:
        return torch.empty((4, 0), dtype=torch.float32)

    height, width = resolution

    egt_raw_events_np = np.stack([
        events['x'], 
        events['y'], 
        events['timestamp'], 
        events['polarity']
    ], axis=1).astype(np.float32)
    
    egt_raw_events_np[:, 0] = egt_raw_events_np[:, 0] / width
    egt_raw_events_np[:, 1] = egt_raw_events_np[:, 1] / height
    if egt_raw_events_np.shape[0] > 0:
        egt_raw_events_np[:, 2] = egt_raw_events_np[:, 2] - egt_raw_events_np[0, 2] 
    egt_raw_events_np[:, 2] = egt_raw_events_np[:, 2] / 1e6
    egt_raw_events_np[:, 3] = egt_raw_events_np[:, 3] * 2 - 1
    
    return torch.from_numpy(egt_raw_events_np.transpose()).to(device)


def create_egt_raw_events(
    events: np.ndarray, 
    resolution: Tuple[int, int], 
    device: torch.device
) -> torch.Tensor:
    if not isinstance(events, np.ndarray) or events.shape[0] == 0:
        return torch.empty((4, 0), dtype=torch.float32)

    height, width = resolution

    curr_np = np.stack([
        events['x'], 
        events['y'], 
        events['timestamp'], 
        events['polarity']
    ], axis=1).astype(np.float32)
    
    curr_np[:, 0] = curr_np[:, 0] / width
    curr_np[:, 1] = curr_np[:, 1] / height
    if curr_np.shape[0] > 0:
        curr_np[:, 2] = curr_np[:, 2] - curr_np[0, 2] 
    curr_np[:, 2] = curr_np[:, 2] / 1e6
    curr_np[:, 3] = curr_np[:, 3] * 2 - 1
    
    return torch.from_numpy(curr_np.transpose()).to(device)

# --- 2. Main dispatcher (Factory) ---

# Use a dictionary to map strings to functions instead of if/elif chains
REPRESENTATION_MAP = {
    "VoxelGrid": create_voxel_grid,
    "VoxelGridComplex": create_complex_voxel_grid,
    "VisEvent": create_visual_frame,
    # You can easily add new representation methods here without modifying the main function below
    # "TimeSurface": create_time_surface,
    "StackedHistogram_from_RVT": create_stacked_histogram_from_rvt,
    "VoxelGrid_from_DEVO": create_voxel_grid_from_devo,
    "EGT_raw_events": create_egt_raw_events,
}

def events_to_representation(
    events: np.ndarray,
    representation_config: Dict[str, Any],
    resolution: Tuple[int, int],
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> torch.Tensor:
    """
    Convert raw event stream into a specified representation according to the config.
    This is a factory function that dispatches to specific representation creation functions.

    Args:
        events (np.ndarray): structured event array in aedat4 format.
        representation_config (Dict[str, Any]): config dict containing 'type' and 'params'.
            e.g., {'type': 'VoxelGrid', 'params': {'num_bins': 5}}
        resolution (Tuple[int, int]): sensor resolution (height, width).
        device (torch.device): computation device.

    Returns:
        torch.Tensor: generated event representation.
    """
    repr_type = representation_config.get("type")
    if repr_type not in REPRESENTATION_MAP:
        raise ValueError(f"Unknown representation type: '{repr_type}'. "
                         f"Available types: {list(REPRESENTATION_MAP.keys())}")

    # Extract specific parameters required by the representation method
    params = representation_config.get("params", {})
    
    # Get the corresponding function from the map
    creation_function = REPRESENTATION_MAP[repr_type]
    
    # Call the function with common and specific parameters
    return creation_function(events=events, resolution=resolution, device=device, **params)


# --- 3. Example usage ---
if __name__ == '__main__':
    # Simulate some fake data
    mock_event_dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('timestamp', '<i8'), ('polarity', '?')])
    mock_events = np.array([
        (10, 20, 1000, True),
        (15, 25, 1500, False),
        (10, 20, 2000, True),
        (30, 40, 3000, True),
    ], dtype=mock_event_dtype)
    
    sensor_resolution = (260, 346)
    
    print("--- Testing VoxelGrid ---")
    config_voxel = {
        "type": "VoxelGrid",
        "params": {
            "num_bins": 5,
            "temporal_bilinear": True
        }
    }
    voxel_grid = events_to_representation(mock_events, config_voxel, sensor_resolution)
    print(f"Output shape: {voxel_grid.shape}")
    print(f"Output device: {voxel_grid.device}")
    assert voxel_grid.shape == (2 * 5, 260, 346)

    print("\n--- Testing VoxelGridComplex ---")
    config_complex = {
        "type": "VoxelGridComplex",
        "params": {
            "num_bins": 3, # different parameter
        }
    }
    complex_voxel = events_to_representation(mock_events, config_complex, sensor_resolution)
    print(f"Output shape: {complex_voxel.shape}")
    assert complex_voxel.shape == (3, 260, 346)
    
    print("\n--- Testing VisEvent ---")
    config_vis = {"type": "VisEvent"}
    vis_frame = events_to_representation(mock_events, config_vis, sensor_resolution)
    print(f"Output shape: {vis_frame.shape}")
    print(f"Output type: {vis_frame.dtype}")
    assert vis_frame.shape == (260, 346, 3)