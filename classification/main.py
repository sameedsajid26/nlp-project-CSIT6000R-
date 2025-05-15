# main.py
import numpy as np
import torch
import random
import argparse
from data_utils import load_dataset, print_dataset_info
from models import get_model
from prompts import get_prompt_fn
from evaluation import evaluate_predictions, save_results
from visualization import plot_confusion_matrix, plot_accuracy_comparison
from config import (
    MATCHED_DATA_PATH, MISMATCHED_DATA_PATH, 
    GPT2_MODEL_NAME, T5_MODEL_NAME, 
    SEED, PROMPT_TYPE, RESULTS_DIR
)
from pathlib import Path

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
def parse_args():
    parser = argparse.ArgumentParser(description="NLI Evaluation")
    parser.add_argument("--gpt2", action="store_true", help="Run GPT-2 evaluation")
    parser.add_argument("--t5", action="store_true", help="Run T5 evaluation")
    parser.add_argument("--matched", action="store_true", help="Evaluate on matched dataset")
    parser.add_argument("--mismatched", action="store_true", help="Evaluate on mismatched dataset")
    parser.add_argument("--prompt", type=str, choices=["zero_shot", "few_shot", "cot"], 
                        default=PROMPT_TYPE, help="Prompt type to use")
    parser.add_argument("--sample", type=int, default=None, 
                        help="Number of samples to use (for testing)")
    args = parser.parse_args()
    
    # Default to running everything if no specific flags are set
    if not (args.gpt2 or args.t5):
        args.gpt2 = args.t5 = True
    if not (args.matched or args.mismatched):
        args.matched = args.mismatched = True
    
    return args

def main():
    args = parse_args()
    
    print("=== NLI Evaluation ===")
    print(f"Using {args.prompt} prompting strategy")
    
    # Create results directory
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    # Load datasets
    datasets = {}
    if args.matched:
        datasets["matched"] = load_dataset(MATCHED_DATA_PATH)
        if args.sample:
            indices = random.sample(range(len(datasets["matched"])), 
                                    min(args.sample, len(datasets["matched"])))
            datasets["matched"] = datasets["matched"].select(indices)
        print_dataset_info("Matched", datasets["matched"])
    
    if args.mismatched:
        datasets["mismatched"] = load_dataset(MISMATCHED_DATA_PATH)
        if args.sample:
            indices = random.sample(range(len(datasets["mismatched"])), 
                                    min(args.sample, len(datasets["mismatched"])))
            datasets["mismatched"] = datasets["mismatched"].select(indices)
        print_dataset_info("Mismatched", datasets["mismatched"])
    
    # Get prompt function
    prompt_fn = get_prompt_fn(args.prompt)
    
    results = {}
    
    # Run GPT-2 evaluation
    if args.gpt2:
        print("\n=== Evaluating GPT-2 ===")
        gpt2_model = get_model("gpt2", GPT2_MODEL_NAME).load()
        
        for dataset_name, dataset in datasets.items():
            print(f"\nEvaluating on {dataset_name} dataset...")
            predictions = gpt2_model.predict_batch(dataset, prompt_fn)
            results[f"gpt2_{dataset_name}"] = evaluate_predictions(predictions, dataset)
            save_results(
                results[f"gpt2_{dataset_name}"], 
                f"gpt2_{args.prompt}", 
                dataset_name
            )
            plot_confusion_matrix(
                results[f"gpt2_{dataset_name}"]["confusion_matrix"],
                ["entailment", "contradiction", "neutral"],
                f"GPT-2 on {dataset_name} dataset",
                f"gpt2_{args.prompt}_{dataset_name}_cm.png"
            )
    
    # Run T5 evaluation
    if args.t5:
        print("\n=== Evaluating T5 ===")
        t5_model = get_model("t5", T5_MODEL_NAME).load()
        
        for dataset_name, dataset in datasets.items():
            print(f"\nEvaluating on {dataset_name} dataset...")
            predictions = t5_model.predict_batch(dataset, prompt_fn)
            results[f"t5_{dataset_name}"] = evaluate_predictions(predictions, dataset)
            save_results(
                results[f"t5_{dataset_name}"], 
                f"t5_{args.prompt}", 
                dataset_name
            )
            plot_confusion_matrix(
                results[f"t5_{dataset_name}"]["confusion_matrix"],
                ["entailment", "contradiction", "neutral"],
                f"T5 on {dataset_name} dataset",
                f"t5_{args.prompt}_{dataset_name}_cm.png"
            )
    
    # Plot accuracy comparison if we have results for both models and datasets
    if (args.gpt2 and args.t5 and args.matched and args.mismatched):
        models = ["GPT-2", "T5"]
        matched_acc = [
            results["gpt2_matched"]["accuracy"],
            results["t5_matched"]["accuracy"]
        ]
        mismatched_acc = [
            results["gpt2_mismatched"]["accuracy"],
            results["t5_mismatched"]["accuracy"]
        ]
        plot_accuracy_comparison(models, matched_acc, mismatched_acc)
    
    print("\n=== Evaluation Complete ===")

if __name__ == "__main__":
    main()