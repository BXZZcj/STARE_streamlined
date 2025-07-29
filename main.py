import json
from omegaconf import OmegaConf
from argparse import ArgumentParser
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

from stare.data import *
from stare.models import *
from stare.core.stare import STARE


def get_experiment_id(config: Dict) -> str:
    """Get or generate a unique experiment ID."""
    exp_id = config.get("experiment_id", "auto")
    if exp_id is None or str(exp_id).lower() == "auto":
        # If not specified or set to 'auto', generate an ID based on the current timestamp
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(exp_id)

def setup_results_path(config: Dict, experiment_id: str, sequence_name: str) -> Path:
    """Construct and create the final output path for the current sequence evaluation."""
    results_config = config.get("results", {})
    
    root_dir_template = results_config.get("output_dir", "experiments/{experiment_id}")
    seq_dir_template = results_config.get("sequence_subdir", "{sequence_name}")
    
    # Fill placeholders
    root_dir = Path(root_dir_template.format(experiment_id=experiment_id))
    final_path = root_dir / seq_dir_template.format(sequence_name=sequence_name)
    
    # Create directory
    final_path.mkdir(parents=True, exist_ok=True)
    return final_path

def generate_metric_name(metric_config: Dict[str, Any]) -> str:
    base_name = metric_config['type']
    params = metric_config.get('params')

    if params:
        sorted_params = sorted(params.items())        
        param_strings = [f"{key}={value}" for key, value in sorted_params]
        param_part = "_".join(param_strings)        
        return f"{base_name}@{param_part}"
    
    return base_name


def main(args):
    # 1. Load config directly
    config = OmegaConf.load(args.config_path)
    experiment_id = get_experiment_id(config)

    # 2. Prepare data
    dataset = eval(config["dataset"]["class_name"])(dataset_path = config["dataset"]["path"])

    # 3. Prepare model
    # (Here we can use a factory pattern to dynamically create models based on config)
    model = eval(config["model"]["class_name"])(
        config = config,
        evs_height = config["dataset"]["evs_height"], 
        evs_width = config["dataset"]["evs_width"], 
        checkpoint_path = config["model"]["checkpoint"]
    )

    # 4. Initialize and run STARE evaluator
    stare = STARE(model, config["stare_params"])
    
    all_sequences_eval_results = defaultdict(list)

    # 5. Evaluate each sequence
    for sequence_name in dataset.get_sequence_list():
        print(f"\n--- Processing sequence: {sequence_name} ---")
        
        # a. Run evaluation
        event_stream = dataset.get_sequence_events(sequence_name)
        ground_truth = dataset.get_sequence_gt(sequence_name)
        outputs_with_timestamps = stare.run(event_stream, ground_truth)
        seq_eval_values = stare.evaluate(outputs_with_timestamps, ground_truth)
        
        # b. Save results for each sequence
        seq_result_path = setup_results_path(config=config, experiment_id=experiment_id, sequence_name=sequence_name)
        
        # Save predictions
        with open(seq_result_path / "predictions.json", "w") as f:
            for output in outputs_with_timestamps:
                output["output"] = output["output"].tolist()
            json.dump(outputs_with_timestamps, f, indent=4)
        
        metrics_config_list = config["stare_params"]["metrics"]
        seq_eval_results = {}
        for metric_config, metric_value in zip(metrics_config_list, seq_eval_values):
            # Generate a unique, descriptive key for the metric
            metric_name = generate_metric_name(metric_config)
            
            # Save to the sequence evaluation result dictionary
            seq_eval_results[metric_name] = metric_value
            
            # Add to the overall collection
            all_sequences_eval_results[metric_name].append(metric_value)
            
        # Save the sequence evaluation result file
        with open(seq_result_path / "results.yaml", "w") as f:
            yaml.dump(seq_eval_results, f, indent=2)
        
        print(f"  - Sequence '{sequence_name}' results saved.")

    # 6. Calculate and save overall evaluation result
    print("\n--- All sequences processed, calculating overall evaluation result... ---")
    
    # a. Calculate the average value for each metric
    overall_eval_result = {}
    for metric_name, value_list in all_sequences_eval_results.items():
        average_value = float(np.mean(value_list))
        overall_eval_result[f"{metric_name}"] = average_value

    # b. Determine the save path (the parent directory of results_path, i.e. the root directory of the experiment)
    # We can find the parent directory by taking the results_path of the last sequence
    experiment_root_path = seq_result_path.parent
    overall_eval_result_filepath = experiment_root_path / "overall_eval_result.yaml"

    # c. Save the overall evaluation result file
    with open(overall_eval_result_filepath, 'w') as f:
        yaml.dump(overall_eval_result, f, indent=2, sort_keys=False)
        
    print(f"--- Experiment {experiment_id} completed! Overall evaluation result saved to: {overall_eval_result_filepath} ---")
    print("\nOverall evaluation result:")
    print(yaml.dump(overall_eval_result, indent=2, sort_keys=False))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="src/stare/configs/demo_devo.yaml")
    args = parser.parse_args()
    
    main(args)