import numpy as np
from typing import List, Tuple, Dict, Any

def calculate_iou_for_bbox(matched_pairs: List[Tuple[List[float], List[float]]]) -> float:
    if not matched_pairs:
        return 0.0  # No matched pairs, return 0

    iou_scores = []
    
    for pred_bbox, gt_bbox in matched_pairs:
        # --- 1. Obtain the coordinates of each bounding box [x1, y1, x2, y2] ---
        # Predicted bounding box
        pred_x1 = pred_bbox[0]
        pred_y1 = pred_bbox[1]
        pred_x2 = pred_bbox[0] + pred_bbox[2]
        pred_y2 = pred_bbox[1] + pred_bbox[3]
        
        # Ground truth bounding box
        gt_x1 = gt_bbox[0]
        gt_y1 = gt_bbox[1]
        gt_x2 = gt_bbox[0] + gt_bbox[2]
        gt_y2 = gt_bbox[1] + gt_bbox[3]

        # --- 2. Calculate the coordinates of the intersection area ---
        inter_x1 = max(pred_x1, gt_x1)
        inter_y1 = max(pred_y1, gt_y1)
        inter_x2 = min(pred_x2, gt_x2)
        inter_y2 = min(pred_y2, gt_y2)

        # --- 3. Calculate the area of the intersection area ---
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        intersection_area = inter_width * inter_height

        # --- 4. Calculate the area of each bounding box ---
        pred_area = pred_bbox[2] * pred_bbox[3]
        gt_area = gt_bbox[2] * gt_bbox[3]
        
        # --- 5. Calculate the area of the union ---
        union_area = pred_area + gt_area - intersection_area

        # --- 6. Calculate IoU ---
        if union_area > 0:
            iou = intersection_area / union_area
        else:
            iou = 0.0

        iou_scores.append(iou)
        
    # Calculate the average IoU and return
    return float(np.mean(iou_scores))


METRIC_MAP = {
    "calculate_iou_for_bbox": calculate_iou_for_bbox,
    # You can easily add new metric functions here
    # "Recall": calculate_recall,
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
