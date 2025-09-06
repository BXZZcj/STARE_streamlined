﻿import json
from omegaconf import OmegaConf
from argparse import ArgumentParser
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import torch

import os
import random

def set_seed(seed: int):
    """
    Set all random seeds to ensure reproducibility.
    """
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"All random seeds set to: {seed}")
set_seed(42)


from stare.data import *
from stare.models import *
from stare.core.stare import STARE


def get_experiment_id(config: Dict, cli_exp_id: str = None) -> str:
    """
    Get or generate a unique experiment ID, with command-line override.
    
    Priority:
    1. Command-line argument (`cli_exp_id`)
    2. Value from config file (`config["experiment_id"]`)
    3. Auto-generated timestamp
    """
    if cli_exp_id:
        return str(cli_exp_id)
    exp_id = config.get("experiment_id", "auto")
    if exp_id is None or str(exp_id).lower() == "auto":
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
    experiment_id = get_experiment_id(config, args.experiment_id)

    # 2. Prepare data
    dataset = eval(config["dataset"]["class_name"])(dataset_path = config["dataset"]["path"])

    # 3. Prepare model
    # (Here we can use a factory pattern to dynamically create models based on config)
    if args.mode in ["run", "all"]:
        model = eval(config["model"]["class_name"])(
            config = config,
            evs_height = config["dataset"]["evs_height"], 
            evs_width = config["dataset"]["evs_width"], 
            checkpoint_path = config["model"]["checkpoint"]
        )
    else:
        model = None

    # 4. Initialize and run STARE evaluator
    stare = STARE(model, dataset, config["stare_params"])
    
    all_sequences_eval_results = defaultdict(list)

    # 5. Evaluate each sequence
    for sequence_name in dataset.get_sequence_list():
        print(f"\n--- Processing sequence: {sequence_name} ---")
        
        seq_result_path = setup_results_path(config=config, experiment_id=experiment_id, sequence_name=sequence_name)
        ground_truth = dataset.get_sequence_gt(sequence_name)
        outputs_with_timestamps = None

        if args.mode in ["run", "all"]:
            print(f"  - [Run Mode] Generating predictions...")
            outputs_with_timestamps = stare.run(sequence_name, ground_truth)
            
            # Save predictions
            predictions_path = seq_result_path / "predictions.json"
            with open(predictions_path, "w") as f:
                outputs_for_json = []
                for output in outputs_with_timestamps:
                    item_copy = output.copy()
                    item_copy["output"] = item_copy["output"].tolist()
                    outputs_for_json.append(item_copy)
                json.dump(outputs_for_json, f, indent=4)
            print(f"  - Predictions saved to {predictions_path}")

        if args.mode in ["evaluate", "all"]:
            print(f"  - [Evaluate Mode] Evaluating predictions...")
            
            if args.mode == "evaluate":
                predictions_path = seq_result_path / "predictions.json"
                if not predictions_path.exists():
                    print(f"  - WARNING: Predictions file not found at {predictions_path}. Skipping evaluation for this sequence.")
                    continue

                with open(predictions_path, "r") as f:
                    loaded_outputs = json.load(f)
                outputs_with_timestamps = []
                for output in loaded_outputs:
                    output["output"] = torch.tensor(output["output"])
                    outputs_with_timestamps.append(output)

            if outputs_with_timestamps is None:
                print(f"  - ERROR: No predictions available to evaluate for sequence '{sequence_name}'.")
                continue

            # Save evaluation results
            seq_eval_values = stare.evaluate(outputs_with_timestamps, ground_truth)
            metrics_config_list = config["stare_params"]["metrics"]
            seq_eval_results = {}
            for metric_config, metric_value in zip(metrics_config_list, seq_eval_values):
                metric_name = generate_metric_name(metric_config)
                seq_eval_results[metric_name] = metric_value
                all_sequences_eval_results[metric_name].append(metric_value)
                
            results_yaml_path = seq_result_path / "results.yaml"
            with open(results_yaml_path, "w") as f:
                yaml.dump(seq_eval_results, f, indent=2)
            
            print(f"  - Sequence '{sequence_name}' results saved to {results_yaml_path}")
            
    if args.mode in ["evaluate", "all"]:
        # 6. Calculate and save overall evaluation result
        print("\n--- All sequences processed, calculating overall evaluation result... ---")
        
        if not all_sequences_eval_results:
             print("--- No sequences were evaluated. Skipping overall results. ---")
             return

        overall_eval_result = {}
        for metric_name, value_list in all_sequences_eval_results.items():
            average_value = float(np.mean(value_list))
            overall_eval_result[f"{metric_name}"] = average_value

        experiment_root_path = seq_result_path.parent
        overall_eval_result_filepath = experiment_root_path / "overall_eval_result.yaml"

        with open(overall_eval_result_filepath, 'w') as f:
            yaml.dump(overall_eval_result, f, indent=2, sort_keys=False)
            
        print(f"--- Experiment {experiment_id} completed! Overall evaluation result saved to: {overall_eval_result_filepath} ---")
        print("\nOverall evaluation result:")
        print(yaml.dump(overall_eval_result, indent=2, sort_keys=False))
    else:
        print(f"\n--- Experiment {experiment_id} completed in run-only mode. ---")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="/home/zongyouyu/nc/STARE_streamlined/src/stare/configs/egt_config.yaml",
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="all", 
        choices=["run", "evaluate", "all"],
        help="Execution mode: 'run' to only generate predictions, 'evaluate' to only evaluate existing predictions, 'all' to do both."
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default=None,
        help="Manually specify an experiment ID. This overrides any ID set in the config file."
    )
    args = parser.parse_args()
    
    main(args)