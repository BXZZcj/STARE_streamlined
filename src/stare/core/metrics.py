﻿import numpy as np
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


def calculate_auc_for_bbox(matched_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
    """
    Calculate the Area Under the Curve (AUC) for a single tracking sequence.

    This function receives a list of matched pairs of predicted and ground truth bounding box tensors.
    The AUC is calculated by measuring the proportion of successfully tracked frames over a range of Intersection over Union (IoU) thresholds.

    Parameters:
        matched_pairs (List[Tuple[torch.Tensor, torch.Tensor]]):
            A list of tuples, each containing (pred_bbox_tensor, gt_bbox_tensor).
            The bounding box format is assumed to be [x, y, w, h, ...].

    Returns:
        float: AUC score, a value between 0.0 and 100.0.
    """
    # --- Edge case handling ---
    if not matched_pairs:
        return 0.0

    # --- Step 1: Extract and prepare data from the input list ---
    # Use list comprehension and torch.stack to efficiently create [N, 4] tensors
    pred_boxes = torch.stack([pair[0][:4] for pair in matched_pairs])
    gt_boxes = torch.stack([pair[1][:4] for pair in matched_pairs])
    
    # Ensure tensors are on the same device (e.g., move both to CPU)
    # You can omit this step if you are sure they are already on the same device
    device = pred_boxes.device
    gt_boxes = gt_boxes.to(device)


    # --- Step 2: Calculate IoU for each frame in a vectorized manner ---
    # Convert [x, y, w, h] format to [x1, y1, x2, y2] format
    pred_x1y1 = pred_boxes[:, :2]
    pred_x2y2 = pred_boxes[:, :2] + pred_boxes[:, 2:]
    gt_x1y1 = gt_boxes[:, :2]
    gt_x2y2 = gt_boxes[:, :2] + gt_boxes[:, 2:]

    # Calculate the coordinates of the intersection
    inter_x1y1 = torch.max(pred_x1y1, gt_x1y1)
    inter_x2y2 = torch.min(pred_x2y2, gt_x2y2)

    # Calculate the area of the intersection
    # clamp(min=0) ensures width and height are 0 when bounding boxes do not overlap
    inter_wh = (inter_x2y2 - inter_x1y1).clamp(min=0)
    intersection_area = inter_wh[:, 0] * inter_wh[:, 1]

    # Calculate the area of the union
    pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
    gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]
    union_area = pred_area + gt_area - intersection_area

    # Calculate IoU scores and handle cases where the denominator is zero
    # Add a small value eps to avoid division by zero
    eps = 1e-7
    ious = intersection_area / (union_area + eps)


    # --- Step 3: Define IoU thresholds ---
    # Standard thresholds from 0.0 to 1.0 with a step of 0.05
    thresholds = torch.arange(0.0, 1.05, 0.05, device=device)


    # --- Step 4: Calculate success rate for each threshold (success curve) ---
    # Use broadcasting to compare all ious with all thresholds at once
    # ious shape: [N], unsqueeze(1) -> [N, 1]
    # thresholds shape: [M], unsqueeze(0) -> [1, M]
    # The comparison results in a boolean matrix success_matrix with shape [N, M]
    success_matrix = ious.unsqueeze(1) > thresholds.unsqueeze(0)

    # Calculate the mean along the frame dimension (dim=0) to get the success rate for each threshold
    success_curve = success_matrix.float().mean(dim=0)


    # --- Step 5: Calculate AUC (mean of the success curve) ---
    # Multiply the result by 100 to make it range from 0 to 100
    auc = success_curve.mean().item() * 100.0

    return auc


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

def calculate_pose_errors(
    matched_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    translation_weight: float = 1.0
) -> float:
    """
    Calculates a single, combined error metric for a list of pose pairs.
    The metric is a weighted sum of the mean translational and rotational errors.

    Parameters:
        matched_pairs (list): A list of tuples. Each tuple contains (prediction, ground_truth).
                           - prediction: A 1D tensor of shape (7,) [x, y, z, qx, qy, qz, qw].
                           - ground_truth: A 1D tensor of shape (7,) [x, y, z, qx, qy, qz, qw].
        translation_weight (float): A weight to balance the translational error (in meters)
                                    against the rotational error (in degrees). 
                                    A weight of 10 means 0.1m error is considered as
                                    severe as a 1-degree error.

    Returns:
        float: A single combined error score. Lower is better.
    """
    if not matched_pairs:
        return 0.0

    # --- 1. Stack tensors for batch processing ---
    predictions = torch.stack([pair[0] for pair in matched_pairs])
    ground_truths = torch.stack([pair[1] for pair in matched_pairs])

    device = predictions.device
    ground_truths = ground_truths.to(device=device, dtype=torch.float32)
    predictions = predictions.to(dtype=torch.float32)

    # --- 2. Calculate Mean Translational Error ---
    pos_pred = predictions[:, :3]
    pos_gt = ground_truths[:, :3]
    translation_errors = torch.norm(pos_pred - pos_gt, dim=1)
    mean_translation_error = torch.mean(translation_errors).item()

    # --- 3. Calculate Mean Rotational Error ---
    quat_pred = predictions[:, 3:]
    quat_gt = ground_truths[:, 3:]
    
    quat_pred = quat_pred / torch.norm(quat_pred, dim=1, keepdim=True)
    quat_gt = quat_gt / torch.norm(quat_gt, dim=1, keepdim=True)

    quat_gt_inv = quat_gt.clone()
    quat_gt_inv[:, :3] *= -1 

    w1, x1, y1, z1 = quat_pred[:, 3], quat_pred[:, 0], quat_pred[:, 1], quat_pred[:, 2]
    w2, x2, y2, z2 = quat_gt_inv[:, 3], quat_gt_inv[:, 0], quat_gt_inv[:, 1], quat_gt_inv[:, 2]
    
    q_rel_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    angular_distance_rad = 2 * torch.acos(torch.clamp(torch.abs(q_rel_w), -1.0, 1.0))
    
    rotation_errors_deg = angular_distance_rad * (180.0 / np.pi)
    mean_rotation_error_deg = torch.mean(rotation_errors_deg).item()

    # --- 4. Combine errors into a single metric ---
    combined_error = (mean_translation_error * translation_weight) + mean_rotation_error_deg
    
    return float(combined_error)
    

METRIC_MAP = {
    "calculate_iou_for_bbox": calculate_iou_for_bbox,
    "calculate_auc_for_bbox": calculate_auc_for_bbox,
    # You can easily add new metric functions here
    # "Recall": calculate_recall,
    "calculate_mIoU_for_detection": calculate_mIoU_for_detection,
    "calculate_pose_errors": calculate_pose_errors,
}

def calculate_metric(
    matched_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    metric_configs: List[Dict[str, Any]]
) -> List:
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