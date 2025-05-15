# evaluation.py
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import os
from pathlib import Path
from config import RESULTS_DIR

def evaluate_predictions(predictions, dataset):
    """Evaluate predictions against gold labels"""
    labels = [item["gold_label"] for item in dataset]
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions)
    conf_matrix = confusion_matrix(
        labels, predictions, 
        labels=["entailment", "contradiction", "neutral"]
    )
    
    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": conf_matrix,
        "predictions": predictions,
        "labels": labels
    }

def save_results(results, model_name, dataset_name):
    """Save evaluation results to files"""
    # Create results directory if it doesn't exist
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    # Save predictions to file
    with open(f"{RESULTS_DIR}/{model_name}_{dataset_name}_predictions.json", "w") as f:
        json.dump({
            "accuracy": results["accuracy"],
            "predictions": results["predictions"],
            "labels": results["labels"]
        }, f)
    
    # Save report
    with open(f"{RESULTS_DIR}/{model_name}_{dataset_name}_report.txt", "w") as f:
        f.write(f"Accuracy: {results['accuracy']:.4f}\n\n")
        f.write(results["report"])
    
    print(f"{model_name} on {dataset_name} - Accuracy: {results['accuracy']:.4f}")
    print(f"Results saved to {RESULTS_DIR}/{model_name}_{dataset_name}_")

def analyze_examples(dataset, predictions, gold_labels, n=5):
    """Analyze specific examples with predictions"""
    # Find examples of different types
    correct_indices = [i for i, (p, g) in enumerate(zip(predictions, gold_labels)) if p == g]
    wrong_indices = [i for i, (p, g) in enumerate(zip(predictions, gold_labels)) if p != g]
    
    # Select random examples from each category (convert to regular Python int)
    if correct_indices:
        random_correct = np.random.choice(correct_indices, min(n, len(correct_indices)), replace=False)
        # Convert numpy int64 to regular Python int
        random_correct = [int(idx) for idx in random_correct]
    else:
        random_correct = []
        
    if wrong_indices:
        random_wrong = np.random.choice(wrong_indices, min(n, len(wrong_indices)), replace=False)
        # Convert numpy int64 to regular Python int
        random_wrong = [int(idx) for idx in random_wrong]
    else:
        random_wrong = []
    
    print("\n=== Correctly Classified Examples ===")
    for idx in random_correct:
        example = dataset[idx]
        print(f"\nExample {idx}:")
        print(f"Premise: {example['sentence1']}")
        print(f"Hypothesis: {example['sentence2']}")
        print(f"Gold label: {gold_labels[idx]}")
        print(f"Prediction: {predictions[idx]}")
    
    print("\n=== Incorrectly Classified Examples ===")
    for idx in random_wrong:
        example = dataset[idx]
        print(f"\nExample {idx}:")
        print(f"Premise: {example['sentence1']}")
        print(f"Hypothesis: {example['sentence2']}")
        print(f"Gold label: {gold_labels[idx]}")
        print(f"Prediction: {predictions[idx]}")