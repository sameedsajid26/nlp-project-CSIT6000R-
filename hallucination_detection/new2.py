import pandas as pd
import numpy as np
import torch
import json
import os
import random
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

# Function to load models
def load_models():
    print("Loading models...")
    
    # Load GPT-2
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load FLAN-T5-Small
    flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    
    print("Models loaded successfully!")
    
    return {
        "gpt2": (gpt2_model, gpt2_tokenizer),
        "flan-t5": (flan_model, flan_tokenizer)
    }

# Function to load the dataset
def load_hallucination_dataset(num_samples=None):
    print("Loading dataset...")
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="evaluation")
    
    # Prepare the data in a more accessible format
    data = []
    for item in dataset:
        wiki_reference = item['wiki_bio_text']
        for sent, label in zip(item['gpt3_sentences'], item['annotation']):
            # Convert annotation to binary
            binary_label = 1 if label in ["minor_inaccurate", "major_inaccurate"] else 0
            data.append({
                'reference': wiki_reference,
                'sentence': sent,
                'original_label': label,
                'binary_label': binary_label
            })
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    if num_samples and num_samples < len(df):
        df = df.sample(num_samples, random_state=42)
    
    print(f"Dataset loaded with {len(df)} sentence pairs")
    print(f"Label distribution: {df['binary_label'].value_counts().to_dict()}")
    return df

# Function to truncate text to fit model limits
def truncate_text(text, max_length=500):
    # Simple truncation strategy - keep the beginning of the text
    if len(text) > max_length:
        return text[:max_length]
    return text

# NLI prompting function for GPT-2
def gpt2_nli_prompt(model, tokenizer, premise, hypothesis, device="cpu", max_length=500):
    # Truncate texts to avoid exceeding model's maximum length
    premise = truncate_text(premise, max_length)
    hypothesis = truncate_text(hypothesis, max_length//10)  # Hypothesis is usually shorter
    
    # Create prompts for entailment, contradiction, and neutral
    templates = {
        "entailment": f"Premise: {premise}\nHypothesis: {hypothesis}\nThe hypothesis is entailed by the premise.",
        "contradiction": f"Premise: {premise}\nHypothesis: {hypothesis}\nThe hypothesis contradicts the premise.",
        "neutral": f"Premise: {premise}\nHypothesis: {hypothesis}\nThe hypothesis is neutral to the premise."
    }
    
    scores = {}
    for label, template in templates.items():
        inputs = tokenizer(template, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Calculate loss as a proxy for likelihood
        loss = outputs.loss
        scores[label] = -loss.item()  # Negative loss = higher likelihood
    
    # Normalize scores to probabilities
    total = sum(np.exp(list(scores.values())))
    probs = {k: np.exp(v) / total for k, v in scores.items()}
    
    return probs

# Improved NLI prediction function for FLAN-T5
def flan_t5_nli_prompt(model, tokenizer, premise, hypothesis, device="cpu", max_length=500):
    # Truncate texts to avoid exceeding model's maximum length
    premise = truncate_text(premise, max_length)
    hypothesis = truncate_text(hypothesis, max_length//10)  # Hypothesis is usually shorter
    
    # Use a more direct prompt format
    prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nQuestion: Is the hypothesis entailed by, contradicted by, or neutral to the premise?\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=20)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
    
    # Log the response for debugging
    print(f"FLAN-T5 response: '{response}'")
    
    # Parse the response more carefully
    probs = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
    
    if any(term in response for term in ["entail", "entailed", "entailment", "yes"]):
        probs["entailment"] = 1.0
    elif any(term in response for term in ["contradict", "contradiction", "no"]):
        probs["contradiction"] = 1.0
    elif any(term in response for term in ["neutral", "neither"]):
        probs["neutral"] = 1.0
    else:
        # If unsure, assign probabilities based on keywords
        for key in probs:
            if key in response:
                probs[key] = 1.0
        
        # If still unsure, use a random assignment for this example
        if sum(probs.values()) == 0:
            print(f"Warning: Unable to parse response '{response}', using random assignment")
            rand_key = random.choice(list(probs.keys()))
            probs[rand_key] = 1.0
    
    return probs

# Alternative direct hallucination detection for FLAN-T5
def flan_t5_direct_hallucination(model, tokenizer, reference, sentence, device="cpu", max_length=500):
    # Truncate texts to avoid exceeding model's maximum length
    reference = truncate_text(reference, max_length)
    sentence = truncate_text(sentence, max_length//10)
    
    # Create a more explicit prompt
    prompt = f"Reference text: {reference}\n\nStatement to verify: {sentence}\n\nQuestion: Based on the reference text, is the statement factual or non-factual?\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=20)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
    
    # Log the response for debugging
    print(f"Direct hallucination response: '{response}'")
    
    # Analyze the response
    if any(term in response for term in ["factual", "accurate", "correct", "true", "supported"]):
        return 0  # Factual
    elif any(term in response for term in ["non-factual", "inaccurate", "incorrect", "false", "not supported", "hallucination"]):
        return 1  # Non-factual
    else:
        # If the response is ambiguous, default to factual (to balance the bias we observed)
        print(f"Warning: Ambiguous response '{response}', defaulting to factual")
        return 0

# Calibrated mapping from NLI predictions to hallucination detection
def calibrated_nli_to_hallucination(nli_probs):
    # If we're highly confident about entailment, consider factual
    # If we're confident about contradiction or neutral, consider non-factual
    # This adds a bias toward factual predictions to counter the observed bias
    
    if nli_probs["entailment"] > 0.4:  # Lower the threshold for entailment
        return 0  # Factual
    else:
        return 1  # Non-factual

# Function to evaluate model performance
def evaluate_model(predictions, ground_truth):
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, zero_division=0)
    recall = recall_score(ground_truth, predictions, zero_division=0)
    f1 = f1_score(ground_truth, predictions, zero_division=0)
    
    # Calculate confusion matrix safely
    cm = confusion_matrix(ground_truth, predictions, labels=[0, 1])
    
    # Now we can safely unpack because we specified the labels
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }

# Function to run in batches
def run_hallucination_detection_in_batches(model_name="flan-t5", method="direct", batch_size=50, device="cpu"):
    # Load models
    models = load_models()
    model, tokenizer = models[model_name]
    model.to(device)
    
    # Load full dataset
    full_df = load_hallucination_dataset(num_samples=None)
    
    # Initialize results
    all_predictions = []
    all_ground_truth = full_df['binary_label'].tolist()
    batch_metrics_list = []
    
    # Create results directory if it doesn't exist
    results_dir = "hallucination_detection_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Configure logging for debugging
    log_file = f"{results_dir}/{model_name}_{method}_responses.log"
    with open(log_file, "w") as f:
        f.write(f"Starting hallucination detection with {model_name} using {method} method\n")
    
    # Process in batches
    for i in range(0, len(full_df), batch_size):
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(full_df) + batch_size - 1)//batch_size}")
        batch_df = full_df.iloc[i:i+batch_size]
        
        # Run inference on batch
        batch_predictions = []
        
        # For debugging, only process the first few examples with verbose output
        verbose_logging = i == 0
        
        for j, (_, row) in enumerate(tqdm(batch_df.iterrows(), total=len(batch_df))):
            premise = row['reference']
            hypothesis = row['sentence']
            
            # Only log detailed responses for the first few examples
            if j < 5 and verbose_logging:
                print("\n" + "="*80)
                print(f"Example {j+1}:")
                print(f"Reference: {premise[:100]}...")
                print(f"Sentence: {hypothesis}")
                print(f"True label: {row['binary_label']} ({row['original_label']})")
            
            if method == "nli":
                if model_name == "gpt2":
                    nli_probs = gpt2_nli_prompt(model, tokenizer, premise, hypothesis, device)
                    prediction = calibrated_nli_to_hallucination(nli_probs)
                else:  # flan-t5
                    # Only enable verbose output for the first few examples
                    if j < 5 and verbose_logging:
                        print("Processing with verbose logging...")
                        nli_probs = flan_t5_nli_prompt(model, tokenizer, premise, hypothesis, device)
                    else:
                        # Redirect print output to suppress logs
                        import sys
                        original_stdout = sys.stdout
                        sys.stdout = open(os.devnull, 'w')
                        try:
                            nli_probs = flan_t5_nli_prompt(model, tokenizer, premise, hypothesis, device)
                        finally:
                            sys.stdout.close()
                            sys.stdout = original_stdout
                    
                    prediction = calibrated_nli_to_hallucination(nli_probs)
            else:  # Direct hallucination detection
                # Only enable verbose output for the first few examples
                if j < 5 and verbose_logging:
                    print("Processing with verbose logging...")
                    prediction = flan_t5_direct_hallucination(model, tokenizer, premise, hypothesis, device)
                else:
                    # Redirect print output to suppress logs
                    import sys
                    original_stdout = sys.stdout
                    sys.stdout = open(os.devnull, 'w')
                    try:
                        prediction = flan_t5_direct_hallucination(model, tokenizer, premise, hypothesis, device)
                    finally:
                        sys.stdout.close()
                        sys.stdout = original_stdout
            
            if j < 5 and verbose_logging:
                print(f"Prediction: {prediction}")
                print("="*80)
            
            batch_predictions.append(prediction)
        
        # Add batch predictions to total predictions
        all_predictions.extend(batch_predictions)
        
        # Show intermediate results
        batch_metrics = evaluate_model(batch_predictions, batch_df['binary_label'].tolist())
        batch_metrics_list.append(batch_metrics)
        
        print(f"\nIntermediate results for batch {i//batch_size + 1}:")
        for metric, value in batch_metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        # Save intermediate results
        batch_df_with_predictions = batch_df.copy()
        batch_df_with_predictions['prediction'] = batch_predictions
        batch_df_with_predictions.to_csv(
            f"{results_dir}/{model_name}_{method}_batch_{i//batch_size + 1}.csv", 
            index=False
        )
    
    # Evaluate overall performance
    overall_metrics = evaluate_model(all_predictions, all_ground_truth)
    
    print(f"\nOverall results for {model_name} using {method} method:")
    for metric, value in overall_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Add predictions to dataframe
    full_df['prediction'] = all_predictions
    
    # Save all metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_data = {
        "model": model_name,
        "method": method,
        "timestamp": timestamp,
        "overall_metrics": overall_metrics,
        "batch_metrics": batch_metrics_list
    }
    
    with open(f"{results_dir}/{model_name}_{method}_metrics_{timestamp}.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
    
    return overall_metrics, all_predictions, full_df

# Function to display confusion matrix and examples
def analyze_results(predictions, df, num_examples=5):
    df['prediction'] = predictions
    
    # Count prediction types
    true_pos = df[(df['binary_label'] == 1) & (df['prediction'] == 1)]
    false_pos = df[(df['binary_label'] == 0) & (df['prediction'] == 1)]
    true_neg = df[(df['binary_label'] == 0) & (df['prediction'] == 0)]
    false_neg = df[(df['binary_label'] == 1) & (df['prediction'] == 0)]
    
    print("\nConfusion Matrix:")
    print(f"True Positives: {len(true_pos)}")
    print(f"False Positives: {len(false_pos)}")
    print(f"True Negatives: {len(true_neg)}")
    print(f"False Negatives: {len(false_neg)}")
    
    # Display examples of errors
    if len(false_pos) > 0:
        print("\nFalse Positive Examples (Predicted non-factual but actually factual):")
        for i, (_, row) in enumerate(false_pos.sample(min(num_examples, len(false_pos))).iterrows()):
            print(f"Example {i+1}:")
            print(f"Reference: {row['reference'][:100]}...")
            print(f"Sentence: {row['sentence']}")
            print(f"Original Label: {row['original_label']}")
            print("-" * 80)
    
    if len(false_neg) > 0:
        print("\nFalse Negative Examples (Predicted factual but actually non-factual):")
        for i, (_, row) in enumerate(false_neg.sample(min(num_examples, len(false_neg))).iterrows()):
            print(f"Example {i+1}:")
            print(f"Reference: {row['reference'][:100]}...")
            print(f"Sentence: {row['sentence']}")
            print(f"Original Label: {row['original_label']}")
            print("-" * 80)
    
    # Save analysis results
    analysis_data = {
        "confusion_matrix": {
            "true_positives": len(true_pos),
            "false_positives": len(false_pos),
            "true_negatives": len(true_neg),
            "false_negatives": len(false_neg)
        },
        "class_distribution": {
            "predicted_factual": len(true_neg) + len(false_neg),
            "predicted_non_factual": len(true_pos) + len(false_pos),
            "actual_factual": len(true_neg) + len(false_pos),
            "actual_non_factual": len(true_pos) + len(false_neg)
        }
    }
    
    results_dir = "hallucination_detection_results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{results_dir}/analysis_{timestamp}.json", "w") as f:
        json.dump(analysis_data, f, indent=4)

# Main execution
if __name__ == "__main__":
    # Choose model and method
    model_name = "flan-t5"  # "gpt2" or "flan-t5"
    method = "direct"  # "nli" or "direct" - Let's try the direct method
    batch_size = 50
    device = "cpu"  # Use "cuda" if GPU is available
    
    # Test with a small sample first to see if we fixed the issue
    test_samples = 10
    print(f"Testing with {test_samples} samples first...")
    
    # Load a small sample
    test_df = load_hallucination_dataset(num_samples=test_samples)
    
    # Manually run a few examples to debug
    models = load_models()
    model, tokenizer = models[model_name]
    
    print("\nTesting direct hallucination detection on a few examples:")
    for i, (_, row) in enumerate(test_df.iterrows()):
        if i >= 3:  # Only test the first 3 examples
            break
        
        premise = row['reference']
        hypothesis = row['sentence']
        
        print(f"\nExample {i+1}:")
        print(f"Reference: {premise[:100]}...")
        print(f"Sentence: {hypothesis}")
        print(f"True label: {row['binary_label']} ({row['original_label']})")
        
        result = flan_t5_direct_hallucination(model, tokenizer, premise, hypothesis)
        print(f"Prediction: {result}")
        print("-" * 80)
    
    # Create results directory
    results_dir = "hallucination_detection_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Log start time
    start_time = datetime.now()
    print(f"Started at: {start_time}")
    
    # Run on all samples with batch processing
    print(f"Running on full dataset with batch processing using {model_name} and {method} method...")
    metrics, predictions, df = run_hallucination_detection_in_batches(
        model_name=model_name, 
        method=method,
        batch_size=batch_size,
        device=device
    )
    
    # Log end time
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Finished at: {end_time}")
    print(f"Total duration: {duration}")
    
    # Analyze the results
    analyze_results(predictions, df)
    
    # Save final results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_file = f"{results_dir}/{model_name}_{method}_full_results_{timestamp}.csv"
    df.to_csv(final_results_file, index=False)
    print(f"Full results saved to {final_results_file}")
    
    # Save execution metadata
    metadata = {
        "model": model_name,
        "method": method,
        "batch_size": batch_size,
        "device": device,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration.total_seconds()
    }
    
    with open(f"{results_dir}/execution_metadata_{timestamp}.json", "w") as f:
        json.dump(metadata, f, indent=4)