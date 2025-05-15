import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from models.models_cls import *

def load_models():
    # Load T5 base model for prompting
    t5_model_name = "t5-base"
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
    
    # Load GPT-2 model for prompting
    gpt2_model_name = "gpt2"
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
    
    # Load pre-trained NLI model for baseline
    nli_model_name = "roberta-large-mnli"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    t5_model.to(device)
    gpt2_model.to(device)
    nli_model.to(device)
    
    return {
        't5': (t5_model, t5_tokenizer), 
        'gpt2': (gpt2_model, gpt2_tokenizer),
        'nli': (nli_model, nli_tokenizer)
    }

# Load models
models = load_models()
# Load the dataset
dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="evaluation")

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame({
    'wiki_bio_text': dataset['wiki_bio_text'],
    'gpt3_sentences': dataset['gpt3_sentences'],
    'annotations': dataset['annotation']
})

# print(df.columns)

# Flatten the data so each row is a single sentence with its annotation
flattened_data = []
for idx, row in df.iterrows():
    wiki_text = row['wiki_bio_text']
    for sent, annot in zip(row['gpt3_sentences'], row['annotations']):
        # Convert to binary classification: 0 for Factual, 1 for Non-Factual (both minor and major)
        binary_label = 0 if annot == "accurate" else 1
        flattened_data.append({
            'premise': wiki_text,
            'hypothesis': sent,
            'original_label': annot,
            'binary_label': binary_label
        })

flat_df = pd.DataFrame(flattened_data)
print(f"Total number of sentences to evaluate: {len(flat_df)}")
# print(flat_df.head())

# For efficiency, let's limit to a subset for initial testing
# Comment this out to run on the full dataset
flat_df = flat_df.sample(100, random_state=42)

# Run predictions with T5
print("Starting T5 predictions...")
t5_results = []
for idx, row in tqdm(flat_df.iterrows(), total=len(flat_df), desc="T5 Predictions"):
    try:
        nli_label, binary_pred, response = predict_nli_with_t5_prompting(
            row['premise'], row['hypothesis'], 
            models['t5'][0], models['t5'][1]
        )
        t5_results.append({
            'idx': idx,
            'nli_label': nli_label,
            'binary_pred': binary_pred,
            'response': response
        })
    except Exception as e:
        print(f"Error with T5 on idx {idx}: {e}")
        t5_results.append({
            'idx': idx,
            'nli_label': "error",
            'binary_pred': 1,  # Default to non-factual for errors
            'response': str(e)
        })

# Run predictions with GPT-2
print("Starting GPT-2 predictions...")
gpt2_results = []
for idx, row in tqdm(flat_df.iterrows(), total=len(flat_df), desc="GPT-2 Predictions"):
    try:
        # Truncate long texts for GPT-2 (which has a context limit)
        premise = row['premise'][:500]  # Truncate very long premises
        hypothesis = row['hypothesis'][:100]  # Truncate very long hypotheses
        
        nli_label, binary_pred, perplexities = predict_nli_with_gpt2_prompting(
            premise, hypothesis, 
            models['gpt2'][0], models['gpt2'][1]
        )
        gpt2_results.append({
            'idx': idx,
            'nli_label': nli_label,
            'binary_pred': binary_pred,
            'perplexities': perplexities
        })
    except Exception as e:
        print(f"Error with GPT-2 on idx {idx}: {e}")
        gpt2_results.append({
            'idx': idx,
            'nli_label': "error",
            'binary_pred': 1,  # Default to non-factual for errors
            'perplexities': {}
        })

# Run predictions with pre-trained NLI model as baseline
print("Starting NLI baseline predictions...")
baseline_results = []
for idx, row in tqdm(flat_df.iterrows(), total=len(flat_df), desc="Baseline NLI Predictions"):
    try:
        nli_label, binary_pred, probs = predict_with_pretrained_nli(
            row['premise'], row['hypothesis'], 
            models['nli'][0], models['nli'][1]
        )
        baseline_results.append({
            'idx': idx,
            'nli_label': nli_label,
            'binary_pred': binary_pred,
            'probs': probs
        })
    except Exception as e:
        print(f"Error with baseline on idx {idx}: {e}")
        baseline_results.append({
            'idx': idx,
            'nli_label': "error",
            'binary_pred': 1,  # Default to non-factual for errors
            'probs': []
        })