import numpy as np
import torch
from typing import List, Tuple, Dict, Any

def calculate_iou_for_bbox(matched_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
    """
    Calculates the mean IoU for a list of matched prediction and ground truth pairs.
    This version is optimized for torch.Tensor inputs.
    
    Parameters:
        matched_pairs (list): A list of tuples, where each tuple is (pred_bbox_tensor, gt_bbox_tensor).
                              Both are 1D tensors, e.g., shape=(6,) for pred, shape=(5,) for gt.
                              Bbox format is [x, y, w, h, ...].
                              
    Returns:
        float: The mean IoU of all matched pairs.
    """
    if not matched_pairs:
        return 0.0

    # 1. Directly stack the tensors from the list of tuples
    #    This is more efficient than creating intermediate python lists
    #    torch.stack creates a new dimension, so we get (N, 4) directly
    pred_tensor = torch.stack([pair[0][:4] for pair in matched_pairs])
    gt_tensor = torch.stack([pair[1][:4] for pair in matched_pairs])

    # 2. Call the vectorized intersection_over_union function
    iou_matrix = intersection_over_union(pred_tensor, gt_tensor)

    # 3. Extract the diagonal elements (IoU between matched pairs)
    iou_scores = torch.diag(iou_matrix)

    # 4. Calculate the mean of the IoU scores
    mean_iou = torch.mean(iou_scores).item()

    return float(mean_iou)


def intersection_over_union(boxes_preds: torch.Tensor, boxes_labels: torch.Tensor) -> torch.Tensor:
    """
    Calculates intersection over union for every pred bbox vs every ground truth bbox.
    
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (N, 4) xywh format
        boxes_labels (tensor): Ground Truth Bounding Boxes (M, 4) xywh format
        
    Returns:
        tensor: Intersection over union for all examples (N, M)
    """
    # Convert xywh to xyxy
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    
    # Get intersection coordinates
    x1 = torch.max(box1_x1, box2_x1.T)
    y1 = torch.max(box1_y1, box2_y1.T)
    x2 = torch.min(box1_x2, box2_x2.T)
    y2 = torch.min(box1_y2, box2_y2.T)

    # Calculate intersection area
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Calculate union area
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area.T - intersection
    
    epsilon = 1e-6
    return intersection / (union + epsilon)

def calculate_mIoU_for_detection(matched_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
    """
    Calculates the mean Intersection over Union (mIoU) of the best matched pairs.
    This version is optimized for torch.Tensor inputs.

    Parameters:
        matched_pairs (list): A list of tuples. Each tuple corresponds to an image and contains:
                              - A predictions tensor of shape (num_preds, 6)
                              - A ground_truths tensor of shape (num_gts, 6)

    Returns:
        float: The mean IoU over all best-matched pairs in the dataset.
    """
    
    all_best_ious = []

    for predictions, ground_truths in matched_pairs:
        # predictions shape: (num_preds, 6) -> [x, y, w, h, conf, class_id]
        # ground_truths shape: (num_gts, 5) -> [x, y, w, h, class_id]
        
        # Skip if there are no ground truths (nothing to match against)
        if ground_truths is None or ground_truths.shape[0] == 0:
            continue
            
        # If there are no predictions, all GTs are "misses" (IoU=0)
        if predictions is None or predictions.shape[0] == 0:
            all_best_ious.extend([0.0] * ground_truths.shape[0])
            continue

        num_gts = ground_truths.shape[0]
        
        # --- Vectorized Inner Loop ---
        # Calculate IoU matrix for all predictions vs all ground truths in the image
        # Shape: (num_preds, num_gts)
        iou_matrix = intersection_over_union(predictions[:, :4], ground_truths[:, :4])

        # Create a class matching matrix
        # Shape: (num_preds, num_gts)
        # It's True where pred_class == gt_class
        class_match_matrix = predictions[:, 5].unsqueeze(1) == ground_truths[:, 5].unsqueeze(0)

        # Apply the class mask to the IoU matrix.
        # Where classes don't match, IoU becomes 0.
        iou_matrix_same_class_only = iou_matrix * class_match_matrix

        if iou_matrix_same_class_only.shape[0] == 0:
            # This case happens if there are GTs but no predictions at all
            best_ious_for_image = torch.zeros(num_gts)
        else:
            # For each ground truth (column), find the max IoU from any prediction (row)
            # This finds the best matching prediction for each GT
            best_ious_for_image, _ = torch.max(iou_matrix_same_class_only, dim=0)
        
        all_best_ious.extend(best_ious_for_image.tolist())
    
    if not all_best_ious:
        return 0.0

    mean_iou = sum(all_best_ious) / len(all_best_ious)
    
    return float(mean_iou)



METRIC_MAP = {
    "calculate_iou_for_bbox": calculate_iou_for_bbox,
    # You can easily add new metric functions here
    # "Recall": calculate_recall,
    "calculate_mIoU_for_detection": calculate_mIoU_for_detection,
}

def calculate_metric(
    matched_pairs: List[Tuple[List[float], List[float]]],
    metric_configs: List[Dict[str, Any]]
) -> List[float]:
    metric_results = []
    for metric_config in metric_configs:
        metric_type = metric_config.get("type")
        if metric_type not in METRIC_MAP:
            raise ValueError(f"Unknown metric type: '{metric_type}'. "
                            f"Available types: {list(METRIC_MAP.keys())}")

        # Extract specific parameters required for this metric method
        params = metric_config.get("params", {})
        
        # Get the corresponding function from the dispatch table
        metric_function = METRIC_MAP[metric_type]
        if params:
            metric_result:float = metric_function(matched_pairs=matched_pairs, **params)
        else:
            metric_result:float = metric_function(matched_pairs=matched_pairs)
        
        metric_results.append(metric_result)
        
    return metric_results
