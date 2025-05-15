# test_sample.py
import numpy as np
import torch
import random
from data_utils import load_dataset, print_dataset_info
from models import get_model
from prompts import get_prompt_fn
from evaluation import evaluate_predictions, analyze_examples
from visualization import plot_confusion_matrix
from config import (
    MATCHED_DATA_PATH, MISMATCHED_DATA_PATH, 
    GPT2_MODEL_NAME, T5_MODEL_NAME, 
    SEED, PROMPT_TYPE
)

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def main():
    print("=== NLI Test Sample Run ===")
    
    # Load a small sample of the data
    print("\nLoading sample data...")
    matched_sample = load_dataset(MATCHED_DATA_PATH, sample=True)
    mismatched_sample = load_dataset(MISMATCHED_DATA_PATH, sample=True)
    
    # Print dataset info
    print_dataset_info("Matched Sample", matched_sample)
    print_dataset_info("Mismatched Sample", mismatched_sample)
    
    # Get prompt function
    prompt_fn = get_prompt_fn(PROMPT_TYPE)
    print(f"\nUsing {PROMPT_TYPE} prompting strategy")
    
    # Test GPT-2
    print("\n=== Testing GPT-2 ===")
    gpt2_model = get_model("gpt2", GPT2_MODEL_NAME).load()
    
    # Make predictions on a single example to test
    example = matched_sample[0]
    print("\nTesting single prediction:")
    print(f"Premise: {example['sentence1']}")
    print(f"Hypothesis: {example['sentence2']}")
    
    gpt2_pred = gpt2_model.predict(example['sentence1'], example['sentence2'], prompt_fn)
    print(f"GPT-2 prediction: {gpt2_pred}")
    print(f"Gold label: {example['gold_label']}")
    
    # Make predictions on the samples
    print("\nRunning batch predictions...")
    gpt2_matched_preds = gpt2_model.predict_batch(matched_sample, prompt_fn)
    gpt2_matched_results = evaluate_predictions(gpt2_matched_preds, matched_sample)
    
    # Analyze examples
    analyze_examples(
        matched_sample, 
        gpt2_matched_results["predictions"], 
        gpt2_matched_results["labels"]
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        gpt2_matched_results["confusion_matrix"],
        ["entailment", "contradiction", "neutral"],
        f"GPT-2 Confusion Matrix (Sample)"
    )
    
    # Test T5
    print("\n=== Testing T5 ===")
    t5_model = get_model("t5", T5_MODEL_NAME).load()
    
    # Make predictions on a single example
    t5_pred = t5_model.predict(example['sentence1'], example['sentence2'], prompt_fn)
    print(f"T5 prediction: {t5_pred}")
    print(f"Gold label: {example['gold_label']}")
    
    # Make predictions on the samples
    print("\nRunning batch predictions...")
    t5_matched_preds = t5_model.predict_batch(matched_sample, prompt_fn)
    t5_matched_results = evaluate_predictions(t5_matched_preds, matched_sample)
    
    # Analyze examples
    analyze_examples(
        matched_sample, 
        t5_matched_results["predictions"], 
        t5_matched_results["labels"]
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        t5_matched_results["confusion_matrix"],
        ["entailment", "contradiction", "neutral"],
        f"T5 Confusion Matrix (Sample)"
    )
    
    print("\n=== Test Complete ===")
    print("If everything looks good, you can run the full evaluation using main.py")

if __name__ == "__main__":
    main()