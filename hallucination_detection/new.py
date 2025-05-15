import pandas as pd
import numpy as np
import torch
import json
import os
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
    return df

# Function to truncate text to fit model limits
def truncate_text(text, max_length=500):
    # Simple truncation strategy - keep the beginning of the text
    if len(text) > max_length:
        return text[:max_length]
    return text

def gpt2_nli_prompt(
    model,
    tokenizer,
    premise,
    hypothesis,
    device="cpu",
    max_length=500
):
    """
    Runs a pseudo-NLI prompt on GPT-2. We compute the likelihood
    of each label ("entailment", "contradiction", "neutral")
    given a base prompt that ends with 'The correct label is:'.
    Returns a dictionary of probabilities for each class, where
    we compute p(label | prompt) via GPT-2 log-likelihood.
    """

    # Helper function to truncate text
    def truncate_text(text, max_len=500):
        return text[:max_len] if len(text) > max_len else text

    # Truncate texts to stay within a safe limit
    premise = truncate_text(premise, max_length)
    hypothesis = truncate_text(hypothesis, max_length // 10)

    # Base prompt asking for the correct label
    # We provide the three possible options in the question.
    base_prompt = (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Possible labels: entailment, contradiction, neutral.\n"
        f"The correct label is:"
    )

    # Our candidate labels
    candidate_labels = ["entailment", "contradiction", "neutral"]

    # We'll store log-likelihood scores here
    label_scores = {}

    # Move model to correct device if not already
    model = model.to(device)
    model.eval()

    # Tokenize the base prompt
    # We'll then append each candidate label to the base prompt
    base_enc = tokenizer(base_prompt, return_tensors="pt")
    base_enc = {k: v.to(device) for k, v in base_enc.items()}

    # We'll record the token IDs for the base prompt:
    # This helps us mask out the base portion so we only measure
    # the log-prob of the appended label tokens.
    base_len = base_enc["input_ids"].shape[1]

    for label in candidate_labels:
        # Full text: base prompt + a space + candidate label
        # Good to add a space so it doesn't jam onto the previous token.
        full_text = base_prompt + " " + label

        # Encode
        inputs = tokenizer(full_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # We set the entire sequence as labels, but we only want
        # the log-likelihood from the label portion. We'll handle this
        # by ignoring the base prompt portion in calculating the total loss.
        with torch.no_grad():
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        # By default, GPT-2 calculates the average cross-entropy loss
        # for all tokens in the sequence. We want the average log-prob
        # just for the last portion (the appended label).
        #
        # A simpler workaround in many zero-shot examples is to compare
        # the total sequence loss, given that the base prompt is the same
        # across all label candidates, and the only difference is the
        # appended label. The difference in total loss is effectively
        # dominated by the label tokens. So we can compare raw loss
        # values for these short suffixes. If you want a more precise
        # approach, you'll need to manually mask out the base tokens.
        
        # We'll just store a negative sign of the loss so that
        # higher is better:
        label_scores[label] = -loss.item()

    # Convert raw scores to probabilities via softmax
    exps = {k: np.exp(v) for k, v in label_scores.items()}
    total_score = sum(exps.values())
    probs = {k: exps[k] / total_score for k in label_scores}

    return probs

def flan_t5_nli_prompt(
    model,
    tokenizer,
    premise,
    hypothesis,
    device="cpu",
    max_length=500
):
    """
    Runs an NLI prompt on FLAN-T5, asking whether the premise
    entails, contradicts, or is neutral to the hypothesis.
    Returns a dictionary of probabilities for each class.
    """
    
    # Helper function to truncate text
    def truncate_text(text, max_len=500):
        return text[:max_len] if len(text) > max_len else text

    # Truncate texts to avoid exceeding model's maximum length
    premise = truncate_text(premise, max_length)
    hypothesis = truncate_text(hypothesis, max_length // 10)  # Hypothesis is usually shorter
    
    # Updated prompt that explicitly requests exactly one word
    prompt = (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n\n"
        "Please answer with exactly one word: 'entailment', 'contradiction', or 'neutral'."
    )

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    # Generate; here we keep generation length small to force short responses
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=6,       # Enough tokens for a single-word answer
            num_beams=3,        # Small beam search
            do_sample=False,    # Disable random sampling
            early_stopping=True
        )

    # Decode the model's output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

    # Prepare a probability dictionary
    probs = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}

    # Strict check for exact match
    if response == "entailment":
        probs["entailment"] = 1.0
    elif response == "contradiction":
        probs["contradiction"] = 1.0
    elif response == "neutral":
        probs["neutral"] = 1.0
    else:
        # Fallback handling: if we didn't detect any of the three tokens, default to "neutral"
        # or consider a custom fallback.
        probs["neutral"] = 1.0

    return probs

# Alternative FLAN-T5 prompt for hallucination detection
def flan_t5_hallucination_prompt(model, tokenizer, reference, sentence, device="cpu", max_length=500):
    # Truncate texts to avoid exceeding model's maximum length
    reference = truncate_text(reference, max_length)
    sentence = truncate_text(sentence, max_length//10)
    
    prompt = f"Reference: {reference}\nStatement: {sentence}\nIs the statement factual or non-factual based on the reference?"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=20)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
    
    # Directly map to hallucination detection
    if any(term in response for term in ["factual", "accurate", "correct", "true"]):
        return 0  # Factual
    else:
        return 1  # Non-factual

# Map NLI predictions to hallucination detection
def nli_to_hallucination(nli_probs):
    # If contradiction or neutral is most likely, consider the sentence non-factual
    # If entailment is most likely, consider the sentence factual
    most_likely = max(nli_probs, key=nli_probs.get)
    
    if most_likely == "entailment":
        return 0  # Factual
    else:
        return 1  # Non-factual

# Updated function to evaluate model performance
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
def run_hallucination_detection_in_batches(model_name="flan-t5", method="nli", batch_size=100, device="cpu"):
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
    
    # Process in batches
    for i in range(0, len(full_df), batch_size):
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(full_df) + batch_size - 1)//batch_size}")
        batch_df = full_df.iloc[i:i+batch_size]
        
        # Run inference on batch
        batch_predictions = []
        
        for _, row in tqdm(batch_df.iterrows(), total=len(batch_df)):
            premise = row['reference']
            hypothesis = row['sentence']
            
            if method == "nli":
                if model_name == "gpt2":
                    nli_probs = gpt2_nli_prompt(model, tokenizer, premise, hypothesis, device)
                    prediction = nli_to_hallucination(nli_probs)
                else:  # flan-t5
                    nli_probs = flan_t5_nli_prompt(model, tokenizer, premise, hypothesis, device)
                    prediction = nli_to_hallucination(nli_probs)
            else:  # Direct hallucination detection
                prediction = flan_t5_hallucination_prompt(model, tokenizer, premise, hypothesis, device)
            
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
    model_name = "gpt2"  # "gpt2" or "flan-t5"
    method = "nli"  # "nli" or "direct"
    batch_size = 50  # Adjusted batch size
    device = "cpu"  # Use "cuda" if GPU is available
    
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