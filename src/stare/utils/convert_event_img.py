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
    xs = torch.from_numpy(events['x']).to(device)
    ys = torch.from_numpy(events['y']).to(device)
    ts = torch.from_numpy(events['timestamp']).to(device)
    ps = torch.from_numpy(events['polarity']).to(device)
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

# --- 2. Main dispatcher (Factory) ---

# Use a dictionary to map strings to functions instead of if/elif chains
REPRESENTATION_MAP = {
    "VoxelGrid": create_voxel_grid,
    "VoxelGridComplex": create_complex_voxel_grid,
    "VisEvent": create_visual_frame,
    # You can easily add new representation methods here without modifying the main function below
    # "TimeSurface": create_time_surface,
    "StackedHistogram_from_RVT": create_stacked_histogram_from_rvt,
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